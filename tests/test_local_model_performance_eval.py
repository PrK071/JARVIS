from __future__ import annotations

from tern.orchestrator.local_model_performance_eval import measure_stream, run_performance_eval


class StreamClient:
    def chat_stream(self, messages, **kwargs):
        assert kwargs["max_tokens"] == 64
        yield {"choices": [{"delta": {"content": "one"}, "finish_reason": None}]}
        yield {
            "choices": [{"delta": {"content": " two"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        }


def test_stream_measurement_captures_ttft_finish_tokens_and_rate():
    result = measure_stream(StreamClient(), [{"role": "user", "content": "x"}])
    assert result.ttft_ms >= 0
    assert result.latency_ms >= result.ttft_ms
    assert result.prompt_tokens == 4 and result.completion_tokens == 2
    assert result.finish_reason == "stop"
    assert result.content_length == 7


def test_performance_report_has_percentiles_and_no_prompt_content():
    report = run_performance_eval(StreamClient(), repeats=3)
    assert report["request_count"] == 3
    assert report["prompt_tokens"] == 12
    assert report["completion_tokens"] == 6
    assert report["resources"] is None
    assert "verification facts" not in repr(report)

