from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import os
import re
import threading
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable


class WebThreatCode(str, Enum):
    ACTIVE_LOCAL_NETWORK_REFERENCE = "ACTIVE_LOCAL_NETWORK_REFERENCE"
    ACTIVE_FILE_SCHEME_REFERENCE = "ACTIVE_FILE_SCHEME_REFERENCE"
    KNOWN_MALICIOUS_HOST = "KNOWN_MALICIOUS_HOST"


@dataclass(frozen=True)
class WebThreatSignal:
    code: WebThreatCode
    context: str
    target_class: str

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code.value,
            "context": self.context,
            "target_class": self.target_class,
        }


@dataclass(frozen=True)
class WebThreatAssessment:
    blocked: bool
    signals: tuple[WebThreatSignal, ...] = ()
    pattern_id: str | None = None
    learned_match: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "blocked": self.blocked,
            "codes": sorted({signal.code.value for signal in self.signals}),
            "signals": [signal.as_dict() for signal in self.signals],
            "pattern_id": self.pattern_id,
            "learned_match": self.learned_match,
        }


class ThreatPatternMemory:
    """Atomic local memory populated only by deterministic high-confidence hits."""

    def __init__(self, path: Path | None):
        self.path = path
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> dict[str, object]:
        empty: dict[str, object] = {
            "version": 1,
            "hosts": {},
            "patterns": {},
        }
        if self.path is None or not self.path.is_file():
            return empty
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return empty
        if not isinstance(value, dict) or value.get("version") != 1:
            return empty
        if not isinstance(value.get("hosts"), dict):
            value["hosts"] = {}
        if not isinstance(value.get("patterns"), dict):
            value["patterns"] = {}
        return value

    def knows_host(self, host: str) -> bool:
        hosts = self._data.get("hosts", {})
        return isinstance(hosts, dict) and host.casefold() in hosts

    def knows_pattern(self, pattern_id: str) -> bool:
        patterns = self._data.get("patterns", {})
        return isinstance(patterns, dict) and pattern_id in patterns

    def observe(
        self,
        *,
        host: str,
        pattern_id: str,
        codes: Iterable[WebThreatCode],
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        normalized_codes = sorted({code.value for code in codes})
        with self._lock:
            hosts = self._data.setdefault("hosts", {})
            patterns = self._data.setdefault("patterns", {})
            if not isinstance(hosts, dict) or not isinstance(patterns, dict):
                return
            host_key = host.casefold()
            host_record = hosts.get(host_key, {})
            pattern_record = patterns.get(pattern_id, {})
            hosts[host_key] = {
                "first_seen": host_record.get("first_seen", now),
                "last_seen": now,
                "hits": int(host_record.get("hits", 0)) + 1,
                "codes": normalized_codes,
            }
            patterns[pattern_id] = {
                "first_seen": pattern_record.get("first_seen", now),
                "last_seen": now,
                "hits": int(pattern_record.get("hits", 0)) + 1,
                "codes": normalized_codes,
            }
            self._save()

    def status(self) -> dict[str, int]:
        hosts = self._data.get("hosts", {})
        patterns = self._data.get("patterns", {})
        return {
            "known_malicious_hosts": len(hosts) if isinstance(hosts, dict) else 0,
            "learned_patterns": len(patterns) if isinstance(patterns, dict) else 0,
        }

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(
            f"{self.path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)


class AdaptiveWebThreatAnalyzer:
    """Static threat analysis with a non-generative, poisoning-resistant memory."""

    _ACTIVE_ATTRIBUTES = {
        "audio": ("src",),
        "embed": ("src",),
        "form": ("action",),
        "frame": ("src",),
        "iframe": ("src",),
        "img": ("src",),
        "input": ("src", "formaction"),
        "link": ("href",),
        "object": ("data",),
        "script": ("src",),
        "source": ("src",),
        "track": ("src",),
        "video": ("src", "poster"),
    }
    _CSS_URL_RE = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)", re.I | re.S)
    _SCRIPT_SINKS = (
        re.compile(
            r"\b(?:fetch|WebSocket|EventSource)\s*\(\s*(['\"])(.*?)\1",
            re.I | re.S,
        ),
        re.compile(
            r"\bnavigator\.sendBeacon\s*\(\s*(['\"])(.*?)\1",
            re.I | re.S,
        ),
        re.compile(
            r"\.open\s*\(\s*(['\"])[A-Z]+\1\s*,\s*(['\"])(.*?)\2",
            re.I | re.S,
        ),
    )

    def __init__(
        self,
        memory_path: Path | None = None,
        *,
        learning_enabled: bool = True,
    ):
        self.learning_enabled = learning_enabled
        self.memory = ThreatPatternMemory(memory_path)

    def preflight(self, raw_url: str) -> WebThreatAssessment:
        try:
            host = (urllib.parse.urlsplit(raw_url).hostname or "").casefold()
        except ValueError:
            return WebThreatAssessment(False)
        if host and self.learning_enabled and self.memory.knows_host(host):
            signal = WebThreatSignal(
                WebThreatCode.KNOWN_MALICIOUS_HOST,
                "request.host",
                "learned_exact_host",
            )
            return WebThreatAssessment(True, (signal,), learned_match=True)
        return WebThreatAssessment(False)

    def inspect_html(self, raw_url: str, data: bytes) -> WebThreatAssessment:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return WebThreatAssessment(False)
        document = BeautifulSoup(
            data.decode("utf-8", errors="replace"), "html.parser"
        )
        references: list[tuple[str, str]] = []
        for tag_name, attributes in self._ACTIVE_ATTRIBUTES.items():
            for tag in document.find_all(tag_name):
                for attribute in attributes:
                    value = tag.get(attribute)
                    if isinstance(value, str):
                        references.append((f"{tag_name}.{attribute}", value))
        for meta in document.find_all("meta"):
            if str(meta.get("http-equiv") or "").casefold() != "refresh":
                continue
            content = str(meta.get("content") or "")
            match = re.search(r"(?:^|;)\s*url\s*=\s*(.+)$", content, re.I)
            if match:
                references.append(("meta.refresh", match.group(1).strip(" '\"")))
        for style in document.find_all("style"):
            references.extend(
                ("style.url", match.group(2))
                for match in self._CSS_URL_RE.finditer(style.get_text(" "))
            )
        for tag in document.find_all(style=True):
            references.extend(
                ("style-attribute.url", match.group(2))
                for match in self._CSS_URL_RE.finditer(str(tag.get("style")))
            )
        for script in document.find_all("script"):
            script_text = script.string or script.get_text(" ")
            for pattern in self._SCRIPT_SINKS:
                for match in pattern.finditer(script_text):
                    references.append(("script.network-sink", match.group(match.lastindex or 1)))

        signals = tuple(
            signal
            for context, reference in references
            if (signal := self._classify_reference(raw_url, context, reference))
            is not None
        )
        if not signals:
            return WebThreatAssessment(False)
        signature = "|".join(
            sorted(
                f"{signal.code.value}:{signal.context}:{signal.target_class}"
                for signal in signals
            )
        )
        pattern_id = hashlib.sha256(signature.encode("utf-8")).hexdigest()
        learned_match = self.memory.knows_pattern(pattern_id)
        assessment = WebThreatAssessment(
            True,
            signals,
            pattern_id=pattern_id,
            learned_match=learned_match,
        )
        host = (urllib.parse.urlsplit(raw_url).hostname or "").casefold()
        if self.learning_enabled and host:
            self.memory.observe(
                host=host,
                pattern_id=pattern_id,
                codes=(signal.code for signal in signals),
            )
        return assessment

    def status(self) -> dict[str, object]:
        return {
            "learning_enabled": self.learning_enabled,
            **self.memory.status(),
        }

    @staticmethod
    def _classify_reference(
        base_url: str,
        context: str,
        raw_reference: str,
    ) -> WebThreatSignal | None:
        reference = html.unescape(raw_reference).strip()
        if not reference or reference.startswith(("#", "data:", "blob:")):
            return None
        try:
            parsed_raw = urllib.parse.urlsplit(reference)
        except ValueError:
            return None
        if parsed_raw.scheme.casefold() == "file":
            return WebThreatSignal(
                WebThreatCode.ACTIVE_FILE_SCHEME_REFERENCE,
                context,
                "local_file",
            )
        absolute = urllib.parse.urljoin(base_url, reference)
        try:
            parsed = urllib.parse.urlsplit(absolute)
        except ValueError:
            return None
        scheme = parsed.scheme.casefold()
        if scheme in {"ws", "wss"}:
            scheme = "http" if scheme == "ws" else "https"
        if scheme not in {"http", "https"}:
            return None
        host = (parsed.hostname or "").casefold().rstrip(".")
        if host == "localhost" or host.endswith((".localhost", ".local")):
            return WebThreatSignal(
                WebThreatCode.ACTIVE_LOCAL_NETWORK_REFERENCE,
                context,
                "localhost",
            )
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
            try:
                if host.isdecimal() and int(host) <= 0xFFFFFFFF:
                    address = ipaddress.ip_address(int(host))
                elif host.lower().startswith("0x"):
                    address = ipaddress.ip_address(int(host, 16))
            except ValueError:
                address = None
        if address is None or not (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
        ):
            return None
        target_class = (
            "loopback"
            if address.is_loopback
            else "link_local_or_metadata"
            if address.is_link_local
            else "private_network"
            if address.is_private
            else "non_public_network"
        )
        return WebThreatSignal(
            WebThreatCode.ACTIVE_LOCAL_NETWORK_REFERENCE,
            context,
            target_class,
        )
