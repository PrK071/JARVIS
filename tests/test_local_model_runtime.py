from __future__ import annotations

import pytest

from tern.orchestrator.client import (
    ServerError,
    ServerTimeoutError,
    ServerUnavailableError,
)
from tern.orchestrator.local_model_runtime import (
    LocalModelRuntimeError,
    OpenAICompatibleLocalRuntime,
    RuntimeDescriptor,
    RuntimeFailureCode,
)


class FakeClient:
    def __init__(self, response=None, *, error=None, health=None, props=None):
        self.response = response or {
            "choices": [
                {
                    "message": {"content": '{"answer":"ok"}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 5},
        }
        self.error = error
        self.health_value = health or {"status": "ok"}
        self.props_value = props or {
            "model_path": r"D:\models\candidate.gguf",
            "model_ftype": "Q2_0",
            "build_info": "b10437-abcd",
            "default_generation_settings": {"n_ctx": 8192},
        }
        self.calls = []

    def health(self):
        if self.error and self.error[0] == "health":
            raise self.error[1]
        return self.health_value

    def props(self):
        if self.error and self.error[0] == "props":
            raise self.error[1]
        return self.props_value

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.error and self.error[0] == "chat":
            raise self.error[1]
        return self.response


def runtime(client=None, *, sink=None, expected_build="b10437"):
    return OpenAICompatibleLocalRuntime(
        client or FakeClient(),
        RuntimeDescriptor(
            provider="candidate",
            model="candidate-model",
            runtime="llama.cpp-mainline-vulkan",
            expected_model_path=r"D:\models\candidate.gguf",
            expected_runtime_version=expected_build,
        ),
        telemetry_sink=sink,
    )


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"answer": {"type": "string", "enum": ["ok"]}},
    "required": ["answer"],
}


def test_health_model_and_runtime_information():
    value = runtime()
    assert value.health() == {"status": "ok"}
    assert value.model_info()["context_size"] == 8192
    assert value.runtime_info()["structured_mode"] == "response_format.json_schema"
    assert value.assert_compatible()["runtime_version"] == "b10437-abcd"


def test_structured_output_uses_one_common_strict_contract_and_observes_no_prompt():
    events = []
    value = runtime(sink=events.append)
    result = value.generate_structured(
        [{"role": "user", "content": "sensitive"}],
        schema=SCHEMA,
        schema_name="basic",
        temperature=0.0,
        max_tokens=32,
    )
    assert result.parsed == {"answer": "ok"}
    sent = value.client.calls[0][1]
    assert sent["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "basic", "strict": True, "schema": SCHEMA},
    }
    assert events[0].validator_result == "passed"
    assert "sensitive" not in repr(events[0].as_dict())
    assert events[0].input_tokens == 7 and events[0].output_tokens == 5


@pytest.mark.parametrize(
    ("content", "code"),
    [
        ("prefix {\"answer\":\"ok\"}", RuntimeFailureCode.MALFORMED_JSON),
        ('{"answer":"wrong"}', RuntimeFailureCode.VALIDATOR_FAILURE),
        ('{"answer":"ok","extra":1}', RuntimeFailureCode.VALIDATOR_FAILURE),
    ],
)
def test_malformed_and_schema_violations_are_rejected(content, code):
    client = FakeClient()
    client.response["choices"][0]["message"]["content"] = content
    with pytest.raises(LocalModelRuntimeError) as caught:
        runtime(client).generate_structured(
            [{"role": "user", "content": "x"}],
            schema=SCHEMA,
            schema_name="basic",
            max_tokens=32,
        )
    assert caught.value.code is code


def test_semantic_validator_parity_rejects_structurally_valid_value():
    with pytest.raises(LocalModelRuntimeError) as caught:
        runtime().generate_structured(
            [{"role": "user", "content": "x"}],
            schema=SCHEMA,
            schema_name="basic",
            max_tokens=32,
            semantic_validator=lambda _value: False,
        )
    assert caught.value.code is RuntimeFailureCode.SEMANTIC_FAILURE


def test_truncation_is_distinct_from_malformed_json():
    client = FakeClient()
    client.response["choices"][0]["finish_reason"] = "length"
    with pytest.raises(LocalModelRuntimeError) as caught:
        runtime(client).generate_structured(
            [{"role": "user", "content": "x"}],
            schema=SCHEMA,
            schema_name="basic",
            max_tokens=1,
        )
    assert caught.value.code is RuntimeFailureCode.TRUNCATION


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (ServerTimeoutError("timeout"), RuntimeFailureCode.LATENCY_TIMEOUT),
        (ServerUnavailableError("offline"), RuntimeFailureCode.SERVER_UNAVAILABLE),
        (ServerError("grammar parse failed"), RuntimeFailureCode.GRAMMAR_FAILURE),
        (ServerError("schema conversion failed"), RuntimeFailureCode.SCHEMA_UNSUPPORTED),
        (ServerError("failed to load model"), RuntimeFailureCode.GGUF_UNSUPPORTED),
    ],
)
def test_transport_failure_taxonomy(error, code):
    with pytest.raises(LocalModelRuntimeError) as caught:
        runtime(FakeClient(error=("chat", error))).generate_text(
            [{"role": "user", "content": "x"}], max_tokens=4
        )
    assert caught.value.code is code


def test_model_unavailable_and_runtime_mismatch_are_separate():
    client = FakeClient()
    client.props_value["model_path"] = r"D:\models\other.gguf"
    with pytest.raises(LocalModelRuntimeError) as model_error:
        runtime(client).assert_compatible()
    assert model_error.value.code is RuntimeFailureCode.MODEL_UNAVAILABLE

    client.props_value["model_path"] = r"D:\models\candidate.gguf"
    with pytest.raises(LocalModelRuntimeError) as runtime_error:
        runtime(client, expected_build="b99999").assert_compatible()
    assert runtime_error.value.code is RuntimeFailureCode.RUNTIME_MISMATCH


def test_invalid_evaluator_schema_fails_before_inference():
    value = runtime()
    with pytest.raises(LocalModelRuntimeError) as caught:
        value.generate_structured(
            [{"role": "user", "content": "x"}],
            schema={"type": "not-a-json-schema-type"},
            schema_name="invalid",
            max_tokens=4,
        )
    assert caught.value.code is RuntimeFailureCode.SCHEMA_UNSUPPORTED
    assert value.client.calls == []

