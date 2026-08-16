from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .client import LlamaClient


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


@dataclass(frozen=True)
class StreamMeasurement:
    ttft_ms: float
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    tokens_per_second: float | None
    finish_reason: str | None
    content_length: int


@dataclass(frozen=True)
class ProcessSample:
    timestamp: float
    working_set_bytes: int
    private_bytes: int
    cpu_seconds: float


class WindowsProcessSampler:
    def __init__(self, pid: int, *, interval: float = 0.1):
        self.pid = pid
        self.interval = interval
        self.samples: list[ProcessSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if os.name != "nt":
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if not self.samples:
            return {
                "sample_count": 0,
                "peak_working_set_bytes": None,
                "peak_private_bytes": None,
                "mean_cpu_percent": None,
            }
        elapsed = self.samples[-1].timestamp - self.samples[0].timestamp
        cpu_delta = self.samples[-1].cpu_seconds - self.samples[0].cpu_seconds
        cpu_percent = (
            100 * cpu_delta / elapsed / max(1, os.cpu_count() or 1)
            if elapsed > 0
            else None
        )
        return {
            "sample_count": len(self.samples),
            "peak_working_set_bytes": max(item.working_set_bytes for item in self.samples),
            "peak_private_bytes": max(item.private_bytes for item in self.samples),
            "mean_cpu_percent": cpu_percent,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            value = self._sample_windows()
            if value is not None:
                self.samples.append(value)
            self._stop.wait(self.interval)

    def _sample_windows(self) -> ProcessSample | None:
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        process_vm_read = 0x0010
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information | process_vm_read,
            False,
            self.pid,
        )
        if not handle:
            return None

        class ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        memory = ProcessMemoryCountersEx()
        memory.cb = ctypes.sizeof(memory)
        creation = wintypes.FILETIME()
        exit_time = wintypes.FILETIME()
        kernel = wintypes.FILETIME()
        user = wintypes.FILETIME()
        try:
            memory_ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                handle,
                ctypes.byref(memory),
                memory.cb,
            )
            times_ok = ctypes.windll.kernel32.GetProcessTimes(
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_time),
                ctypes.byref(kernel),
                ctypes.byref(user),
            )
            if not memory_ok or not times_ok:
                return None
            kernel_ticks = (kernel.dwHighDateTime << 32) | kernel.dwLowDateTime
            user_ticks = (user.dwHighDateTime << 32) | user.dwLowDateTime
            return ProcessSample(
                time.time(),
                int(memory.WorkingSetSize),
                int(memory.PrivateUsage),
                (kernel_ticks + user_ticks) / 10_000_000,
            )
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)


def measure_stream(
    client: LlamaClient,
    messages: list[dict[str, Any]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 64,
) -> StreamMeasurement:
    started = time.perf_counter()
    first_content: float | None = None
    content: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    for event in client.chat_stream(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
    ):
        usage.update(event.get("usage") or {})
        for choice in event.get("choices") or []:
            delta = (choice.get("delta") or {}).get("content")
            if delta:
                if first_content is None:
                    first_content = time.perf_counter()
                content.append(str(delta))
            if choice.get("finish_reason") is not None:
                finish_reason = str(choice["finish_reason"])
    finished = time.perf_counter()
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    ttft = (first_content or finished) - started
    generation_seconds = finished - (first_content or finished)
    rate = completion_tokens / generation_seconds if generation_seconds > 0 else None
    return StreamMeasurement(
        ttft_ms=ttft * 1000,
        latency_ms=(finished - started) * 1000,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tokens_per_second=rate,
        finish_reason=finish_reason,
        content_length=sum(len(item) for item in content),
    )


def run_performance_eval(
    client: LlamaClient,
    *,
    repeats: int,
    pid: int | None = None,
) -> dict[str, Any]:
    sampler = WindowsProcessSampler(pid) if pid else None
    if sampler:
        sampler.start()
    records = []
    try:
        for _index in range(repeats):
            records.append(
                measure_stream(
                    client,
                    [
                        {"role": "system", "content": "Answer concisely."},
                        {
                            "role": "user",
                            "content": "List three deterministic verification facts for a code change.",
                        },
                    ],
                    temperature=0.0,
                    max_tokens=64,
                )
            )
    finally:
        resources = sampler.stop() if sampler else None
    ttft = [item.ttft_ms for item in records]
    latency = [item.latency_ms for item in records]
    rates = [item.tokens_per_second for item in records if item.tokens_per_second is not None]
    return {
        "request_count": len(records),
        "ttft_ms": {
            "mean": statistics.fmean(ttft),
            "p50": _percentile(ttft, 0.50),
            "p90": _percentile(ttft, 0.90),
            "p95": _percentile(ttft, 0.95),
            "p99": _percentile(ttft, 0.99),
        },
        "latency_ms": {
            "mean": statistics.fmean(latency),
            "p50": _percentile(latency, 0.50),
            "p90": _percentile(latency, 0.90),
            "p95": _percentile(latency, 0.95),
            "p99": _percentile(latency, 0.99),
        },
        "tokens_per_second": {
            "mean": statistics.fmean(rates) if rates else None,
            "values": rates,
        },
        "prompt_tokens": sum(item.prompt_tokens for item in records),
        "completion_tokens": sum(item.completion_tokens for item in records),
        "resources": resources,
        "records": [asdict(item) for item in records],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local model streaming performance evaluator")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = run_performance_eval(
        LlamaClient(args.endpoint, timeout=args.timeout),
        repeats=args.repeats,
        pid=args.pid,
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

