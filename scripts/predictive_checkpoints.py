"""Journal do evaluator fora do projeto analisado, com flush independente."""
import ctypes
import json
import os
import sys
import time
from pathlib import Path


def own_working_set_bytes():
    if sys.platform != 'win32':
        return None
    from ctypes import wintypes
    class MemoryCounters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('peak', 'working', 'quota_peak_paged',
                'quota_paged', 'quota_peak_nonpaged', 'quota_nonpaged', 'pagefile', 'peak_pagefile')]
    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    function = ctypes.windll.psapi.GetProcessMemoryInfo
    function.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
    function.restype = wintypes.BOOL
    process = ctypes.windll.kernel32.GetCurrentProcess
    process.restype = wintypes.HANDLE
    return int(counters.working) if function(process(), ctypes.byref(counters), counters.cb) else None


class CheckpointJournal:
    def __init__(self, path: Path, snapshots: Path):
        self.path = path.resolve()
        if self.path.is_relative_to(snapshots.resolve()) or path.is_symlink():
            raise ValueError('Journal nao pode modificar snapshots externos.')
        if self.path.exists():
            raise ValueError('Nao sobrescrever checkpoints historicos.')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self.sequence = 0

    def __call__(self, event):
        self.sequence += 1
        row = dict(event, schema_version=1, sequence=self.sequence,
            wall_seconds=time.perf_counter()-self.started,
            working_set_bytes=own_working_set_bytes(),
            model_timing_resolution='REQUEST_INCLUSIVE_QUEUE_AND_GENERATION_UNKNOWN')
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
            handle.flush()
            os.fsync(handle.fileno())


def last_checkpoint(path):
    if not path.exists():
        return None
    result = None
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                break
    return result
