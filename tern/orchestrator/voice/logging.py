from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class VoiceLogger:
    def __init__(
        self,
        path: Path,
        *,
        level: str = "INFO",
        debug_transcripts: bool = False,
    ):
        self.path = path
        self.level = level.upper()
        self.debug_transcripts = debug_transcripts
        self._lock = threading.Lock()

    def write(self, event: str, **values: Any) -> None:
        record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **values,
        }
        if not self.debug_transcripts:
            record.pop("transcript", None)
            record.pop("text", None)
            record.pop("transcript_normalized", None)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + os.linesep)
