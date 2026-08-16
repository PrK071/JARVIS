from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Protocol

from jsonschema import Draft202012Validator, SchemaError, ValidationError

from .client import ServerError, ServerTimeoutError, ServerUnavailableError


class RuntimeFailureCode(str, Enum):
    RUNTIME_UNSUPPORTED = "RUNTIME_UNSUPPORTED"
    GGUF_UNSUPPORTED = "GGUF_UNSUPPORTED"
    SCHEMA_UNSUPPORTED = "SCHEMA_UNSUPPORTED"
    GRAMMAR_FAILURE = "GRAMMAR_FAILURE"
    MALFORMED_JSON = "MALFORMED_JSON"
    TRUNCATION = "TRUNCATION"
    MODEL_NONCOMPLIANCE = "MODEL_NONCOMPLIANCE"
    SERVER_ERROR = "SERVER_ERROR"
    SERVER_UNAVAILABLE = "SERVER_UNAVAILABLE"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    RUNTIME_MISMATCH = "RUNTIME_MISMATCH"
    LATENCY_TIMEOUT = "LATENCY_TIMEOUT"
    VALIDATOR_FAILURE = "VALIDATOR_FAILURE"
    SEMANTIC_FAILURE = "SEMANTIC_FAILURE"


class StructuredMode(str, Enum):
    RESPONSE_FORMAT_JSON_SCHEMA = "response_format.json_schema"


@dataclass(frozen=True)
class RuntimeDescriptor:
    provider: str
    model: str
    runtime: str
    expected_model_path: str | None = None
    expected_runtime_version: str | None = None
    structured_mode: StructuredMode = StructuredMode.RESPONSE_FORMAT_JSON_SCHEMA


@dataclass(frozen=True)
class InferenceObservation:
    request_id: str
    model: str
    runtime: str
    runtime_version: str | None
    schema_id: str | None
    structured_mode: str | None
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    retry: bool
    fallback: bool
    validator_result: str
    semantic_result: str | None
    finish_reason: str | None
    failure_code: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationResult:
    content: str
    parsed: Any | None
    response: Mapping[str, Any]
    observation: InferenceObservation


class LocalModelRuntimeError(RuntimeError):
    def __init__(
        self,
        code: RuntimeFailureCode,
        message: str,
        *,
        observation: InferenceObservation | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.observation = observation


class ChatTransport(Protocol):
    def health(self) -> dict[str, Any]: ...

    def props(self) -> dict[str, Any]: ...

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> dict[str, Any]: ...


class LocalModelRuntime(Protocol):
    def health(self) -> dict[str, Any]: ...

    def model_info(self) -> dict[str, Any]: ...

    def runtime_info(self) -> dict[str, Any]: ...

    def generate_text(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> GenerationResult: ...

    def generate_structured(
        self,
        messages: list[dict[str, Any]],
        *,
        schema: Mapping[str, Any],
        schema_name: str,
        temperature: float,
        max_tokens: int,
        semantic_validator: Callable[[Any], Any] | None = None,
    ) -> GenerationResult: ...


TelemetrySink = Callable[[InferenceObservation], None]


def schema_fingerprint(schema: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        schema,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


class OpenAICompatibleLocalRuntime:
    """One logical Jarvis contract over an OpenAI-compatible local server.

    Model-specific prompt repair and permissive JSON extraction deliberately do
    not belong here. Both control and candidate must pass the same parser,
    Draft 2020-12 validator and optional semantic validator.
    """

    def __init__(
        self,
        client: ChatTransport,
        descriptor: RuntimeDescriptor,
        *,
        telemetry_sink: TelemetrySink | None = None,
    ):
        self.client = client
        self.descriptor = descriptor
        self.telemetry_sink = telemetry_sink

    def health(self) -> dict[str, Any]:
        try:
            value = self.client.health()
        except Exception as exc:
            raise self._transport_error(exc) from exc
        if value.get("status") != "ok":
            raise LocalModelRuntimeError(
                RuntimeFailureCode.SERVER_UNAVAILABLE,
                f"local model health check failed: {value!r}",
            )
        return value

    def model_info(self) -> dict[str, Any]:
        props = self._props()
        return {
            "provider": self.descriptor.provider,
            "model": self.descriptor.model,
            "model_path": props.get("model_path") or props.get("model_alias"),
            "model_type": props.get("model_ftype"),
            "context_size": (
                (props.get("default_generation_settings") or {}).get("n_ctx")
            ),
        }

    def runtime_info(self) -> dict[str, Any]:
        props = self._props()
        return {
            "runtime": self.descriptor.runtime,
            "runtime_version": props.get("build_info"),
            "structured_mode": self.descriptor.structured_mode.value,
            "streaming": True,
            "finish_reason": True,
            "max_tokens": True,
            "temperature": True,
        }

    def assert_compatible(self) -> dict[str, Any]:
        self.health()
        props = self._props()
        model_path = str(props.get("model_path") or props.get("model_alias") or "")
        expected_path = self.descriptor.expected_model_path
        if expected_path and self._normalized_path(model_path) != self._normalized_path(expected_path):
            raise LocalModelRuntimeError(
                RuntimeFailureCode.MODEL_UNAVAILABLE,
                f"runtime loaded {model_path!r}, expected {expected_path!r}",
            )
        build = str(props.get("build_info") or "")
        expected_build = self.descriptor.expected_runtime_version
        if expected_build and expected_build not in build:
            raise LocalModelRuntimeError(
                RuntimeFailureCode.RUNTIME_MISMATCH,
                f"runtime build {build!r} does not match {expected_build!r}",
            )
        return {"health": "ok", "model_path": model_path, "runtime_version": build}

    def generate_text(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> GenerationResult:
        return self._generate(
            messages,
            response_format=None,
            schema=None,
            schema_id=None,
            temperature=temperature,
            max_tokens=max_tokens,
            semantic_validator=None,
        )

    def generate_structured(
        self,
        messages: list[dict[str, Any]],
        *,
        schema: Mapping[str, Any],
        schema_name: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        semantic_validator: Callable[[Any], Any] | None = None,
    ) -> GenerationResult:
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            raise LocalModelRuntimeError(
                RuntimeFailureCode.SCHEMA_UNSUPPORTED,
                f"invalid evaluator schema: {exc.message}",
            ) from exc
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": dict(schema),
            },
        }
        return self._generate(
            messages,
            response_format=response_format,
            schema=schema,
            schema_id=schema_fingerprint(schema),
            temperature=temperature,
            max_tokens=max_tokens,
            semantic_validator=semantic_validator,
        )

    def _generate(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None,
        schema: Mapping[str, Any] | None,
        schema_id: str | None,
        temperature: float,
        max_tokens: int,
        semantic_validator: Callable[[Any], Any] | None,
    ) -> GenerationResult:
        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        runtime_version = self._runtime_version_best_effort()
        try:
            response = self.client.chat(
                messages,
                tools=None,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            failure = self._transport_error(exc)
            observation = self._observation(
                request_id,
                started,
                runtime_version,
                schema_id,
                None,
                None,
                "not_run",
                None,
                None,
                failure.code,
            )
            self._emit(observation)
            raise LocalModelRuntimeError(
                failure.code,
                str(failure),
                observation=observation,
            ) from exc

        try:
            choice = (response.get("choices") or [])[0]
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("message content is not text")
        except (IndexError, KeyError, TypeError) as exc:
            return self._raise_result_error(
                RuntimeFailureCode.SERVER_ERROR,
                "OpenAI-compatible response has no text choice",
                request_id,
                started,
                runtime_version,
                schema_id,
                response,
                None,
                "failed",
                None,
                exc,
            )

        choice = (response.get("choices") or [{}])[0]
        finish_reason = choice.get("finish_reason")
        usage = response.get("usage") or {}
        prompt_tokens = self._integer(usage.get("prompt_tokens"))
        output_tokens = self._integer(usage.get("completion_tokens"))
        if finish_reason == "length":
            return self._raise_result_error(
                RuntimeFailureCode.TRUNCATION,
                "generation stopped at max_tokens",
                request_id,
                started,
                runtime_version,
                schema_id,
                response,
                finish_reason,
                "failed",
                None,
            )

        parsed: Any | None = None
        validator_result = "not_requested"
        semantic_result: str | None = None
        if schema is not None:
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                return self._raise_result_error(
                    RuntimeFailureCode.MALFORMED_JSON,
                    f"structured response is not one JSON value: {exc.msg}",
                    request_id,
                    started,
                    runtime_version,
                    schema_id,
                    response,
                    finish_reason,
                    "failed",
                    None,
                    exc,
                )
            try:
                Draft202012Validator(schema).validate(parsed)
            except ValidationError as exc:
                return self._raise_result_error(
                    RuntimeFailureCode.VALIDATOR_FAILURE,
                    f"structured response violates schema: {exc.message}",
                    request_id,
                    started,
                    runtime_version,
                    schema_id,
                    response,
                    finish_reason,
                    "failed",
                    None,
                    exc,
                )
            validator_result = "passed"
            if semantic_validator is not None:
                try:
                    semantic_value = semantic_validator(parsed)
                    if semantic_value is False:
                        raise ValueError("semantic validator returned false")
                except Exception as exc:
                    return self._raise_result_error(
                        RuntimeFailureCode.SEMANTIC_FAILURE,
                        f"semantic validation failed: {exc}",
                        request_id,
                        started,
                        runtime_version,
                        schema_id,
                        response,
                        finish_reason,
                        validator_result,
                        "failed",
                        exc,
                    )
                semantic_result = "passed"

        observation = self._observation(
            request_id,
            started,
            runtime_version,
            schema_id,
            prompt_tokens,
            output_tokens,
            validator_result,
            semantic_result,
            finish_reason,
            None,
        )
        self._emit(observation)
        return GenerationResult(content, parsed, response, observation)

    def _raise_result_error(
        self,
        code: RuntimeFailureCode,
        message: str,
        request_id: str,
        started: float,
        runtime_version: str | None,
        schema_id: str | None,
        response: Mapping[str, Any],
        finish_reason: str | None,
        validator_result: str,
        semantic_result: str | None,
        cause: Exception | None = None,
    ) -> GenerationResult:
        usage = response.get("usage") or {}
        observation = self._observation(
            request_id,
            started,
            runtime_version,
            schema_id,
            self._integer(usage.get("prompt_tokens")),
            self._integer(usage.get("completion_tokens")),
            validator_result,
            semantic_result,
            finish_reason,
            code,
        )
        self._emit(observation)
        error = LocalModelRuntimeError(code, message, observation=observation)
        if cause is not None:
            raise error from cause
        raise error

    def _observation(
        self,
        request_id: str,
        started: float,
        runtime_version: str | None,
        schema_id: str | None,
        input_tokens: int | None,
        output_tokens: int | None,
        validator_result: str,
        semantic_result: str | None,
        finish_reason: str | None,
        failure_code: RuntimeFailureCode | None,
    ) -> InferenceObservation:
        return InferenceObservation(
            request_id=request_id,
            model=self.descriptor.model,
            runtime=self.descriptor.runtime,
            runtime_version=runtime_version,
            schema_id=schema_id,
            structured_mode=(
                self.descriptor.structured_mode.value if schema_id else None
            ),
            latency_ms=(time.perf_counter() - started) * 1000,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            retry=False,
            fallback=False,
            validator_result=validator_result,
            semantic_result=semantic_result,
            finish_reason=finish_reason,
            failure_code=failure_code.value if failure_code else None,
        )

    def _props(self) -> dict[str, Any]:
        try:
            return self.client.props()
        except Exception as exc:
            raise self._transport_error(exc) from exc

    def _runtime_version_best_effort(self) -> str | None:
        try:
            return str(self.client.props().get("build_info") or "") or None
        except Exception:
            return None

    def _transport_error(self, exc: Exception) -> LocalModelRuntimeError:
        if isinstance(exc, LocalModelRuntimeError):
            return exc
        if isinstance(exc, ServerTimeoutError) or isinstance(exc, TimeoutError):
            return LocalModelRuntimeError(RuntimeFailureCode.LATENCY_TIMEOUT, str(exc))
        if isinstance(exc, ServerUnavailableError):
            return LocalModelRuntimeError(RuntimeFailureCode.SERVER_UNAVAILABLE, str(exc))
        if isinstance(exc, ServerError):
            lowered = str(exc).casefold()
            if "failed to load model" in lowered or "model" in lowered and "load" in lowered:
                code = RuntimeFailureCode.GGUF_UNSUPPORTED
            elif "grammar" in lowered:
                code = RuntimeFailureCode.GRAMMAR_FAILURE
            elif "schema" in lowered:
                code = RuntimeFailureCode.SCHEMA_UNSUPPORTED
            else:
                code = RuntimeFailureCode.SERVER_ERROR
            return LocalModelRuntimeError(code, str(exc))
        return LocalModelRuntimeError(RuntimeFailureCode.SERVER_ERROR, str(exc))

    def _emit(self, observation: InferenceObservation) -> None:
        if self.telemetry_sink is not None:
            self.telemetry_sink(observation)

    @staticmethod
    def _integer(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalized_path(value: str) -> str:
        return value.replace("/", "\\").rstrip("\\").casefold()

