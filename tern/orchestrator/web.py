from __future__ import annotations

import hashlib
import ipaddress
import os
import re
import socket
import ssl
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

from .research import (
    ResearchIntent,
    classify_research_request,
    generate_query_variants,
    normalize_text,
    parse_date,
    score_result,
    validate_opened_source,
)
from .web_threats import AdaptiveWebThreatAnalyzer


class WebError(RuntimeError):
    code = "web_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


class WebAccessDenied(WebError):
    code = "web_access_denied"


class WebTimeout(WebError, TimeoutError):
    code = "web_timeout"


class WebTooLarge(WebError):
    code = "web_response_too_large"


class SearchProviderNotConfigured(WebError):
    code = "search_provider_not_configured"


class SearchAuthenticationFailed(WebError):
    code = "search_authentication_failed"


class SearchHttpError(WebError):
    code = "search_http_error"


class SearchDnsError(WebError):
    code = "search_dns_error"


class SearchTimeout(WebTimeout):
    code = "search_timeout"


class SearchResponseInvalid(WebError):
    code = "search_response_invalid"


@dataclass(frozen=True)
class WebConfig:
    enabled: bool = True
    search_provider: str = "bing_rss"
    search_url: str = "https://www.bing.com/search"
    search_api_key: str | None = None
    safe_search: str = "off"
    threat_analysis_enabled: bool = True
    threat_learning_enabled: bool = True
    threat_memory_path: Path | None = None
    timeout: int = 20
    max_download_bytes: int = 10 * 1024 * 1024
    max_text_chars: int = 65536
    max_pdf_pages: int = 20
    user_agent: str = "TernLocalResearch/1.0"
    allowed_domains: tuple[str, ...] = ()
    blocked_domains: tuple[str, ...] = ()
    query_expansion_enabled: bool = True
    max_query_variants: int = 4
    cross_language_search: bool = True
    default_region: str = "BR"
    min_result_relevance: float = 0.55
    min_source_relevance: float = 0.65
    relevance_top_k: int = 8
    max_research_corrections: int = 2
    max_total_searches: int = 6
    max_total_opens: int = 10


@dataclass(frozen=True)
class FetchResponse:
    url: str
    status: int
    headers: dict[str, str]
    data: bytes
    duration_ms: int = 0


Resolver = Callable[[str], Iterable[str]]
Transport = Callable[[str], FetchResponse]
TraceCallback = Callable[[str, dict[str, Any]], None]
BrowserOpener = Callable[[str], bool]

_LANGUAGE_RE = re.compile(r"^[a-zA-Z]{2,3}(?:-[a-zA-Z]{2})?$")
_SAFE_SEARCH_LEVELS = {"off", "moderate", "strict"}
_TOKEN_RE = re.compile(r"[\wÀ-ÿ]{2,}", re.UNICODE)
_STOPWORDS = {
    "a",
    "ao",
    "as",
    "com",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "na",
    "nas",
    "no",
    "nos",
    "o",
    "os",
    "para",
    "por",
    "que",
    "the",
    "and",
    "for",
    "with",
}


def _detect_windows_default_browser_executable() -> str | None:
    if os.name != "nt":
        return None
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\https\UserChoice",
        ) as key:
            prog_id = str(winreg.QueryValueEx(key, "ProgId")[0])
        with winreg.OpenKey(
            winreg.HKEY_CLASSES_ROOT,
            rf"{prog_id}\shell\open\command",
        ) as key:
            command = str(winreg.QueryValueEx(key, "")[0])
    except (OSError, ImportError, ValueError):
        return None
    quoted = re.match(r'^\s*"([^"]+\.exe)"', command, re.IGNORECASE)
    bare = re.match(r"^\s*([^\s]+\.exe)", command, re.IGNORECASE)
    executable = quoted or bare
    return executable.group(1) if executable else None


@lru_cache(maxsize=1)
def _windows_default_browser_executable() -> str | None:
    """Identify the Windows default browser once for this Jarvis process."""
    return _detect_windows_default_browser_executable()


_BROWSER_EXECUTABLE_NAMES = frozenset(
    {
        "brave.exe",
        "chrome.exe",
        "chromium.exe",
        "firefox.exe",
        "msedge.exe",
        "opera.exe",
        "opera_gx.exe",
        "vivaldi.exe",
        "waterfox.exe",
    }
)


def _windows_running_browser_executable() -> str | None:
    """Return executable for the topmost visible browser window, if any."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        user32.IsWindowVisible.argtypes = [wintypes.HWND]
        user32.IsWindowVisible.restype = wintypes.BOOL
        user32.GetWindowThreadProcessId.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(wintypes.DWORD),
        ]
        user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        found: list[str] = []
        process_query_limited_information = 0x1000

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def inspect_window(window: int, _parameter: int) -> bool:
            if not user32.IsWindowVisible(window):
                return True
            process_id = wintypes.DWORD()
            user32.GetWindowThreadProcessId(window, ctypes.byref(process_id))
            handle = kernel32.OpenProcess(
                process_query_limited_information,
                False,
                process_id.value,
            )
            if not handle:
                return True
            try:
                buffer = ctypes.create_unicode_buffer(32768)
                size = wintypes.DWORD(len(buffer))
                if not kernel32.QueryFullProcessImageNameW(
                    handle,
                    0,
                    buffer,
                    ctypes.byref(size),
                ):
                    return True
                executable = buffer.value
                if Path(executable).name.casefold() in _BROWSER_EXECUTABLE_NAMES:
                    found.append(executable)
                    return False
            finally:
                kernel32.CloseHandle(handle)
            return True

        user32.EnumWindows(inspect_window, 0)
        return found[0] if found else None
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _open_system_browser(url: str) -> bool:
    """Open a URL with the cached default browser, reusing its existing session."""
    executable = (
        _windows_default_browser_executable()
        or _windows_running_browser_executable()
    )
    if executable:
        browser_name = Path(executable).name.casefold()
        arguments = (
            [executable, "-new-tab", url]
            if browser_name in {"firefox.exe", "waterfox.exe"}
            else [executable, url]
        )
        try:
            subprocess.Popen(
                arguments,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                close_fds=True,
            )
            return True
        except OSError:
            pass
    return bool(webbrowser.open(url, new=2, autoraise=True))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_space(value: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", value).strip()


def _normalize_domains(values: Iterable[str]) -> tuple[str, ...]:
    result = []
    for value in values:
        domain = value.strip().lower().rstrip(".")
        if domain.startswith("*."):
            domain = domain[2:]
        if not domain or "/" in domain or ":" in domain:
            raise WebAccessDenied(f"dominio invalido: {value!r}")
        try:
            domain = domain.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise WebAccessDenied(f"dominio invalido: {value!r}") from exc
        if domain not in result:
            result.append(domain)
    return tuple(result)


def _domain_matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


class NetworkPolicy:
    def __init__(
        self,
        allowed_domains: Iterable[str] = (),
        blocked_domains: Iterable[str] = (),
        resolver: Resolver | None = None,
    ):
        self.allowed_domains = _normalize_domains(allowed_domains)
        self.blocked_domains = _normalize_domains(blocked_domains)
        self.resolver = resolver or self._system_resolver

    @staticmethod
    def _system_resolver(host: str) -> Iterable[str]:
        return {
            item[4][0]
            for item in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        }

    def permits_domain(
        self,
        host: str,
        *,
        allowed_domains: Iterable[str] | None = None,
        blocked_domains: Iterable[str] | None = None,
    ) -> bool:
        host = host.lower().rstrip(".")
        allowed = (
            self.allowed_domains
            if allowed_domains is None
            else _normalize_domains(allowed_domains)
        )
        blocked = (
            self.blocked_domains
            if blocked_domains is None
            else _normalize_domains(blocked_domains)
        )
        if any(_domain_matches(host, domain) for domain in blocked):
            return False
        return not allowed or any(_domain_matches(host, domain) for domain in allowed)

    @staticmethod
    def _public_address(value: str) -> bool:
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            return False
        return not (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
        )

    def validate_url(self, raw_url: str, *, resolve_dns: bool = True) -> str:
        if not isinstance(raw_url, str) or len(raw_url) > 8192:
            raise WebAccessDenied("URL invalida")
        try:
            parsed = urllib.parse.urlsplit(raw_url.strip())
            port = parsed.port
        except ValueError as exc:
            raise WebAccessDenied(f"URL invalida: {exc}") from exc
        if parsed.scheme.lower() not in {"http", "https"}:
            raise WebAccessDenied("somente URLs HTTP/HTTPS sao permitidas")
        if parsed.username is not None or parsed.password is not None:
            raise WebAccessDenied("credenciais em URL sao proibidas")
        if not parsed.hostname:
            raise WebAccessDenied("URL sem host")
        if port is not None and port not in {80, 443}:
            raise WebAccessDenied("somente portas 80 e 443 sao permitidas")
        try:
            host = parsed.hostname.lower().rstrip(".").encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise WebAccessDenied("host invalido") from exc
        if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
            raise WebAccessDenied("host local bloqueado")
        if not self.permits_domain(host):
            raise WebAccessDenied(f"dominio bloqueado: {host}")

        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            literal = None
        if literal is not None:
            if not self._public_address(str(literal)):
                raise WebAccessDenied(f"endereco de rede privado bloqueado: {literal}")
        elif resolve_dns:
            try:
                addresses = tuple(self.resolver(host))
            except (OSError, socket.gaierror) as exc:
                raise WebError(f"falha DNS para {host}: {exc}") from exc
            if not addresses:
                raise WebError(f"DNS sem endereco para {host}")
            if any(not self._public_address(address) for address in addresses):
                raise WebAccessDenied(f"DNS de {host} aponta para rede nao publica")

        default_port = (parsed.scheme.lower() == "http" and port == 80) or (
            parsed.scheme.lower() == "https" and port == 443
        )
        display_host = f"[{host}]" if ":" in host else host
        netloc = display_host if port is None or default_port else f"{display_host}:{port}"
        return urllib.parse.urlunsplit(
            (parsed.scheme.lower(), netloc, parsed.path or "/", parsed.query, "")
        )


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, policy: NetworkPolicy):
        super().__init__()
        self.policy = policy

    def redirect_request(
        self,
        request,
        file_pointer,
        code,
        message,
        headers,
        new_url,
    ):
        safe_url = self.policy.validate_url(new_url)
        return super().redirect_request(
            request, file_pointer, code, message, headers, safe_url
        )


class WebClient:
    def __init__(
        self,
        config: WebConfig,
        *,
        resolver: Resolver | None = None,
        transport: Transport | None = None,
        browser_opener: BrowserOpener | None = None,
    ):
        if config.safe_search not in _SAFE_SEARCH_LEVELS:
            raise WebError(
                "safe_search deve ser off, moderate ou strict"
            )
        self.config = config
        self.threat_analyzer = (
            AdaptiveWebThreatAnalyzer(
                config.threat_memory_path,
                learning_enabled=config.threat_learning_enabled,
            )
            if config.threat_analysis_enabled
            else None
        )
        self.policy = NetworkPolicy(
            config.allowed_domains,
            config.blocked_domains,
            resolver,
        )
        self.search_policy = NetworkPolicy((), config.blocked_domains, resolver)
        self.transport = transport
        self.browser_opener = browser_opener or _open_system_browser
        self._request_text = ""
        self._classification: ResearchIntent | None = None
        self._generated_queries: list[str] = []
        self._executed_queries: list[str] = []
        self._result_by_url: dict[str, dict[str, Any]] = {}
        self._search_count = 0
        self._open_count = 0
        self._correction_count = 0
        self._accepted_urls: set[str] = set()
        self._trace_callback: TraceCallback | None = None

    def set_trace_callback(self, callback: TraceCallback | None) -> None:
        self._trace_callback = callback

    def _trace(self, stage: str, **values: Any) -> None:
        if self._trace_callback is not None:
            self._trace_callback(stage, values)

    def begin_research(self, request_text: str) -> None:
        self._request_text = request_text.strip()
        self._classification = None
        self._generated_queries = []
        self._executed_queries = []
        self._result_by_url = {}
        self._search_count = 0
        self._open_count = 0
        self._correction_count = 0
        self._accepted_urls = set()

    def research_status(self) -> dict[str, Any]:
        return {
            "intent": (
                self._classification.as_dict()
                if self._classification is not None
                else None
            ),
            "searches": self._search_count,
            "opens": self._open_count,
            "corrections": self._correction_count,
            "accepted_sources": len(self._accepted_urls),
            "threat_analysis": (
                self.threat_analyzer.status()
                if self.threat_analyzer is not None
                else {"enabled": False}
            ),
        }

    def search(
        self,
        *,
        query: str,
        max_results: int = 8,
        language: str = "pt-BR",
        freshness_days: int | None = None,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
    ) -> dict[str, Any]:
        self._ensure_enabled()
        query = query.strip()
        if not query:
            raise WebError("consulta vazia")
        if not _LANGUAGE_RE.fullmatch(language):
            raise WebError("language deve usar formato como pt-BR ou en")
        allowed = _normalize_domains(allowed_domains or ())
        blocked = _normalize_domains(blocked_domains or ())
        if self._classification is None:
            self._classification = classify_research_request(
                self._request_text or query,
                language=language,
                region=self.config.default_region,
            )
        classification = self._classification
        if classification.ambiguous:
            return {
                "ok": True,
                "provider": self.config.search_provider,
                "query": query,
                "intent": classification.as_dict(),
                "needs_clarification": True,
                "clarification": (
                    "O termo é ambíguo. Especifique inteligência artificial, "
                    "filme, livro, música ou outro significado."
                ),
                "generated_queries": [],
                "executed_queries": [],
                "results": [],
                "rejected_results": [],
                "result_count": 0,
                "searched_at": _now(),
            }
        if not self._generated_queries:
            if self.config.query_expansion_enabled:
                self._generated_queries = generate_query_variants(
                    classification,
                    maximum=self.config.max_query_variants,
                    cross_language=self.config.cross_language_search,
                )
            else:
                self._generated_queries = [query]
        if not self._generated_queries:
            self._generated_queries = [query]
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        received = 0
        http_values: list[dict[str, Any]] = []
        searches_this_call = 0
        for variant in self._generated_queries:
            if variant in self._executed_queries:
                continue
            if self._search_count >= self.config.max_total_searches:
                break
            if (
                searches_this_call > 0
                and self._correction_count
                >= self.config.max_research_corrections
            ):
                break
            parsed_results, response = self._search_provider_once(
                variant,
                language=(
                    "en"
                    if " latest " in f" {variant.casefold()} "
                    else language
                ),
                max_results=max_results,
                freshness_days=freshness_days,
                allowed=allowed,
                blocked=blocked,
            )
            self._search_count += 1
            searches_this_call += 1
            if searches_this_call > 1:
                self._correction_count += 1
            self._executed_queries.append(variant)
            received += len(parsed_results)
            http_values.append(
                {
                    "status": response.status,
                    "content_type": response.headers.get("content-type"),
                    "duration_ms": response.duration_ms,
                    "bytes": len(response.data),
                }
            )
            for item in parsed_results:
                candidate = self._normalize_search_item(
                    item,
                    allowed=allowed,
                    blocked=blocked,
                    search_query=variant,
                )
                if candidate is None:
                    continue
                scored = score_result(candidate, classification)
                key = self._result_key(scored["url"])
                title_key = normalize_text(scored["title"])
                duplicate = any(
                    self._result_key(value["url"]) == key
                    or normalize_text(value["title"]) == title_key
                    for value in [*accepted, *rejected]
                )
                if duplicate:
                    continue
                self._result_by_url[key] = scored
                threshold = self.config.min_result_relevance
                if classification.intent == "general_information":
                    threshold = min(threshold, 0.30)
                topic_is_sufficient = (
                    classification.intent != "news"
                    or scored["topic_score"] >= 0.65
                )
                type_is_sufficient = (
                    classification.intent != "news"
                    or scored["source_type"]
                    in {"news_article", "official_announcement"}
                )
                if (
                    scored["final_score"] >= threshold
                    and topic_is_sufficient
                    and type_is_sufficient
                ):
                    accepted.append(scored)
                else:
                    reason = (
                        "tópico central insuficiente"
                        if not topic_is_sufficient
                        else (
                            "tipo de fonte não atende à notícia"
                            if not type_is_sufficient
                            else (
                                "relevância abaixo de "
                                f"{self.config.min_result_relevance:.2f}"
                            )
                        )
                    )
                    rejected.append(
                        {
                            **scored,
                            "rejection_reason": reason,
                        }
                    )
            if accepted:
                break
        accepted.sort(key=lambda item: item["final_score"], reverse=True)
        rejected.sort(key=lambda item: item["final_score"], reverse=True)
        limit = min(
            max_results,
            self.config.relevance_top_k,
        )
        visible = accepted[:limit]
        for index, item in enumerate(visible, 1):
            item["id"] = index
        return {
            "ok": True,
            "provider": self.config.search_provider,
            "query": query,
            "language": language,
            "freshness_days": freshness_days,
            "intent": classification.as_dict(),
            "generated_queries": list(self._generated_queries),
            "executed_queries": list(self._executed_queries),
            "correction_count": self._correction_count,
            "results_received": received,
            "results": visible,
            "rejected_results": rejected,
            "result_count": len(visible),
            "searched_at": _now(),
            "http": http_values[-1] if http_values else None,
            "http_requests": http_values,
            "limits": {
                "minimum_result_relevance": self.config.min_result_relevance,
                "total_searches": self._search_count,
                "max_total_searches": self.config.max_total_searches,
            },
            "notice": (
                "Resultados foram pontuados; abra fontes aceitas. "
                "Snippets não são evidência."
            ),
        }

    def _search_provider_once(
        self,
        query: str,
        *,
        language: str,
        max_results: int,
        freshness_days: int | None,
        allowed: tuple[str, ...],
        blocked: tuple[str, ...],
    ) -> tuple[list[dict[str, Any]], FetchResponse]:
        effective_query = query
        if allowed:
            domain_query = " OR ".join(f"site:{domain}" for domain in allowed)
            effective_query = f"{query} ({domain_query})"
        if blocked:
            effective_query += " " + " ".join(f"-site:{domain}" for domain in blocked)
        freshness_bucket = self._freshness_bucket(freshness_days)
        provider = self.config.search_provider
        headers: dict[str, str] = {}
        if provider == "bing_rss":
            parameters = {
                "q": effective_query,
                "format": "rss",
                "setlang": language,
                "adlt": self.config.safe_search,
            }
            headers["Accept"] = (
                "application/rss+xml, application/xml, text/xml"
            )
        elif provider == "brave":
            if not self.config.search_api_key:
                raise SearchProviderNotConfigured(
                    "WEB_SEARCH_API_KEY ausente para provedor brave",
                    details={"provider": provider},
                )
            parameters = {
                "q": effective_query,
                "count": max_results,
                "search_lang": language.split("-", 1)[0].lower(),
                "safesearch": self.config.safe_search,
            }
            if freshness_bucket:
                parameters["freshness"] = {
                    "d": "pd",
                    "w": "pw",
                    "m": "pm",
                    "y": "py",
                }[freshness_bucket]
            headers["X-Subscription-Token"] = self.config.search_api_key
            headers["Accept"] = "application/json"
        elif provider == "duckduckgo_html":
            parameters = {
                "q": effective_query,
                "kl": self._duckduckgo_region(language),
                "kp": {
                    "off": "-2",
                    "moderate": "-1",
                    "strict": "1",
                }[self.config.safe_search],
            }
            if freshness_bucket:
                parameters["df"] = freshness_bucket
            headers["Accept"] = "text/html"
        else:
            raise SearchProviderNotConfigured(
                f"WEB_SEARCH_PROVIDER desconhecido: {provider!r}",
                details={"provider": provider},
            )
        provider_url = self.config.search_url
        if (
            provider == "bing_rss"
            and self._classification is not None
            and self._classification.intent == "news"
        ):
            parsed_provider_url = urllib.parse.urlsplit(provider_url)
            if parsed_provider_url.path.rstrip("/") == "/search":
                provider_url = urllib.parse.urlunsplit(
                    (
                        parsed_provider_url.scheme,
                        parsed_provider_url.netloc,
                        "/news/search",
                        parsed_provider_url.query,
                        "",
                    )
                )
        separator = "&" if "?" in provider_url else "?"
        search_url = (
            provider_url
            + separator
            + urllib.parse.urlencode(parameters)
        )
        response = self._fetch(
            search_url,
            policy=self.search_policy,
            headers=headers,
            operation="search",
        )
        return self._parse_provider_response(response), response

    @staticmethod
    def _result_key(url: str) -> str:
        parts = urllib.parse.urlsplit(url)
        query = [
            (key, value)
            for key, value in urllib.parse.parse_qsl(
                parts.query, keep_blank_values=True
            )
            if not key.casefold().startswith("utm_")
            and key.casefold() not in {"fbclid", "gclid", "ref"}
        ]
        return urllib.parse.urlunsplit(
            (
                parts.scheme.casefold(),
                (parts.hostname or "").casefold()
                + (f":{parts.port}" if parts.port else ""),
                parts.path.rstrip("/") or "/",
                urllib.parse.urlencode(sorted(query)),
                "",
            )
        )

    def _normalize_search_item(
        self,
        item: dict[str, Any],
        *,
        allowed: tuple[str, ...],
        blocked: tuple[str, ...],
        search_query: str,
    ) -> dict[str, Any] | None:
        raw_url = str(item.get("url") or "").strip()
        parts = urllib.parse.urlsplit(raw_url)
        if (parts.hostname or "").endswith("bing.com"):
            parameters = urllib.parse.parse_qs(parts.query)
            redirect = (
                parameters.get("url")
                or parameters.get("target")
                or parameters.get("r")
            )
            if redirect:
                raw_url = urllib.parse.unquote(redirect[0])
        raw_url = self._result_key(raw_url)
        try:
            safe_url = self.policy.validate_url(raw_url, resolve_dns=False)
            host = urllib.parse.urlsplit(safe_url).hostname or ""
            if not self.policy.permits_domain(
                host,
                allowed_domains=allowed or None,
                blocked_domains=blocked or None,
            ):
                return None
        except WebError:
            return None
        return {
            "title": _normalize_space(str(item.get("title") or "")),
            "url": safe_url,
            "domain": host,
            "snippet": self._strip_markup(str(item.get("snippet") or "")),
            "published_at": item.get("published_at"),
            "search_query": search_query,
        }

    def _parse_provider_response(
        self, response: FetchResponse
    ) -> list[dict[str, Any]]:
        provider = self.config.search_provider
        content_type = response.headers.get("content-type", "").lower()
        body_preview = self._sanitized_preview(response.data)
        details = {
            "provider": provider,
            "status": response.status,
            "content_type": content_type or None,
            "duration_ms": response.duration_ms,
            "body_preview": body_preview,
        }
        if provider == "bing_rss":
            if "xml" not in content_type and not response.data.lstrip().startswith(
                b"<?xml"
            ):
                raise SearchResponseInvalid(
                    "Bing RSS retornou conteudo nao XML", details=details
                )
            return self._parse_bing_rss(response.data, details)
        if provider == "brave":
            if "json" not in content_type:
                raise SearchResponseInvalid(
                    "Brave Search retornou conteudo nao JSON", details=details
                )
            return self._parse_brave_json(response.data, details)
        if provider == "duckduckgo_html":
            html = self._decode(response)
            if (
                response.status != 200
                or "anomaly.js" in html
                or "challenge-form" in html
                or "Unfortunately, bots use DuckDuckGo too" in html
            ):
                raise SearchResponseInvalid(
                    "DuckDuckGo retornou desafio anti-bot, nao resultados",
                    details=details,
                )
            parsed = self._parse_search_results(html)
            if not parsed and "result__a" not in html:
                raise SearchResponseInvalid(
                    "HTML do DuckDuckGo nao corresponde ao parser",
                    details=details,
                )
            return [
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "published_at": None,
                }
                for title, url, snippet in parsed
            ]
        raise SearchProviderNotConfigured(
            f"provedor desconhecido: {provider!r}",
            details={"provider": provider},
        )

    @staticmethod
    def _parse_bing_rss(
        data: bytes, details: dict[str, Any]
    ) -> list[dict[str, Any]]:
        try:
            root = ET.fromstring(data)
        except ET.ParseError as exc:
            raise SearchResponseInvalid(
                f"XML Bing RSS invalido: {exc}", details=details
            ) from exc
        channel = root.find("channel")
        if root.tag != "rss" or channel is None:
            raise SearchResponseInvalid(
                "estrutura Bing RSS sem rss/channel", details=details
            )
        results = []
        for item in channel.findall("item"):
            title = _normalize_space(item.findtext("title") or "")
            url = _normalize_space(item.findtext("link") or "")
            snippet = WebClient._strip_markup(
                item.findtext("description") or ""
            )
            if not title or not url:
                raise SearchResponseInvalid(
                    "item Bing RSS sem title ou link", details=details
                )
            results.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "published_at": WebClient._normalize_published_at(
                        item.findtext("pubDate")
                    ),
                }
            )
        return results

    @staticmethod
    def _normalize_published_at(value: str | None) -> str | None:
        raw = _normalize_space(value or "")
        if not raw:
            return None
        parsed = parse_date(raw)
        return parsed.isoformat() if parsed is not None else raw

    @staticmethod
    def _parse_brave_json(
        data: bytes, details: dict[str, Any]
    ) -> list[dict[str, Any]]:
        import json

        try:
            payload = json.loads(data.decode("utf-8"))
            values = payload["web"]["results"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise SearchResponseInvalid(
                f"JSON Brave Search invalido: {type(exc).__name__}",
                details=details,
            ) from exc
        if not isinstance(values, list):
            raise SearchResponseInvalid(
                "Brave Search web.results nao e lista", details=details
            )
        results = []
        for item in values:
            if not isinstance(item, dict):
                raise SearchResponseInvalid(
                    "Brave Search retornou item invalido", details=details
                )
            title = _normalize_space(str(item.get("title") or ""))
            url = _normalize_space(str(item.get("url") or ""))
            if not title or not url:
                raise SearchResponseInvalid(
                    "item Brave Search sem title ou url", details=details
                )
            results.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": WebClient._strip_markup(
                        str(item.get("description") or "")
                    ),
                    "published_at": item.get("page_age")
                    or item.get("age")
                    or None,
                }
            )
        return results

    @staticmethod
    def _strip_markup(value: str) -> str:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return _normalize_space(re.sub(r"<[^>]+>", " ", value))
        return _normalize_space(BeautifulSoup(value, "html.parser").get_text(" "))

    def _sanitized_preview(self, data: bytes, limit: int = 500) -> str:
        text = data[:4096].decode("utf-8", errors="replace")
        if self.config.search_api_key:
            text = text.replace(self.config.search_api_key, "<hidden>")
        return _normalize_space(text)[:limit]

    def open(
        self,
        *,
        url: str,
        max_chars: int = 32768,
        page_start: int | None = None,
        page_end: int | None = None,
        _operation: str = "open",
    ) -> dict[str, Any]:
        self._ensure_enabled()
        if self.threat_analyzer is not None:
            preflight = self.threat_analyzer.preflight(url)
            if preflight.blocked:
                self._trace(
                    "WEB_THREAT_ANALYSIS",
                    result="blocked_before_fetch",
                    reason_code="KNOWN_MALICIOUS_HOST",
                )
                raise WebAccessDenied(
                    "host bloqueado pela memoria local de ameacas",
                    details=preflight.as_dict(),
                )
        if self._open_count >= self.config.max_total_opens:
            raise WebError(
                f"limite de {self.config.max_total_opens} fontes abertas atingido"
            )
        self._open_count += 1
        requested_candidate = self._result_by_url.get(self._result_key(url))
        response = self._fetch(url, operation=_operation)
        final_candidate = self._result_by_url.get(
            self._result_key(response.url)
        )
        search_candidate = final_candidate or requested_candidate
        content_type = response.headers.get("content-type", "").lower()
        if self.threat_analyzer is not None and (
            "text/html" in content_type
            or "application/xhtml+xml" in content_type
            or not content_type
        ):
            assessment = self.threat_analyzer.inspect_html(
                response.url, response.data
            )
            if assessment.blocked:
                self._trace(
                    "WEB_THREAT_ANALYSIS",
                    result="blocked_after_fetch",
                    reason_code=",".join(
                        str(code) for code in assessment.as_dict()["codes"]
                    ),
                )
                raise WebAccessDenied(
                    "pagina bloqueada por referencias ativas a recursos locais",
                    details=assessment.as_dict(),
                )
            self._trace(
                "WEB_THREAT_ANALYSIS",
                result="allowed",
                reason_code="NO_HIGH_CONFIDENCE_SIGNAL",
            )
        browser_preflight_limited = (
            _operation == "browser" and not 200 <= response.status < 300
        )
        if browser_preflight_limited:
            document = {
                "title": urllib.parse.urlsplit(response.url).hostname or response.url,
                "text": "",
                "links": [],
                "published_at": None,
                "document_type": "browser_preflight",
            }
            self._trace(
                "BROWSER_PREFLIGHT",
                result="limited",
                reason_code=f"HTTP_{response.status}",
            )
        elif response.data.startswith(b"%PDF-") or "application/pdf" in content_type:
            document = self._extract_pdf(response, page_start, page_end)
        elif (
            not content_type
            or "text/html" in content_type
            or "application/xhtml+xml" in content_type
        ):
            document = self._extract_html(response)
        elif content_type.startswith("text/plain"):
            document = {
                "title": urllib.parse.urlsplit(response.url).path.rsplit("/", 1)[-1]
                or response.url,
                "text": _normalize_space(self._decode(response)),
                "links": [],
                "published_at": None,
                "document_type": "text",
            }
        else:
            raise WebError(
                f"tipo de conteudo nao suportado: {content_type or 'desconhecido'}"
            )
        self._trace(
            "CONTENT_EXTRACTION",
            result="completed",
            reason_code=str(document.get("document_type") or "unknown"),
        )
        full_text = document.pop("text")
        limit = min(max_chars, self.config.max_text_chars)
        text = full_text[:limit]
        published_at = document.get("published_at")
        if published_at is None and search_candidate is not None:
            published_at = search_candidate.get("published_at")
        result = {
            "ok": True,
            "url": response.url,
            "title": document["title"],
            "document_type": document["document_type"],
            "content_type": content_type.split(";", 1)[0] or None,
            "http_status": response.status,
            "fetched_at": _now(),
            "published_at": published_at,
            "bytes": len(response.data),
            "sha256": hashlib.sha256(response.data).hexdigest(),
            "text": text,
            "truncated": len(full_text) > len(text),
            "links": document.get("links", []),
            **{
                key: value
                for key, value in document.items()
                if key
                not in {"title", "document_type", "published_at", "links"}
            },
        }
        if not browser_preflight_limited:
            result["citation"] = {"title": document["title"], "url": response.url}
        if self._classification is not None and (
            self._classification.intent == "news"
            or bool(self._result_by_url)
        ):
            validation = validate_opened_source(
                result,
                self._classification,
                minimum_score=self.config.min_source_relevance,
            )
            result["validation"] = validation
            result["accepted_for_citation"] = validation["relevant"]
            if search_candidate is not None:
                result["search_result_score"] = search_candidate["final_score"]
                result["published_at_source"] = (
                    "page"
                    if document.get("published_at")
                    else "search_result"
                )
            if validation["relevant"]:
                self._accepted_urls.add(self._result_key(response.url))
            else:
                result.pop("citation", None)
                if (
                    self._correction_count
                    < self.config.max_research_corrections
                    and any(
                        query not in self._executed_queries
                        for query in self._generated_queries
                    )
                ):
                    self._correction_count += 1
                    result["corrective_search"] = self.search(
                        query=self._classification.topic,
                        max_results=self.config.relevance_top_k,
                        language=self._classification.language,
                        freshness_days=90
                        if self._classification.requires_freshness
                        else None,
                    )
        return result

    def open_in_browser(self, *, url: str) -> dict[str, Any]:
        """Validate a URL, then launch it even when automation gets an HTTP error."""
        inspected = self.open(url=url, max_chars=4096, _operation="browser")
        final_url = str(inspected["url"])
        preflight_status = int(inspected.get("http_status") or 0) or None
        preflight_limited = bool(
            preflight_status is not None and not 200 <= preflight_status < 300
        )
        self._trace(
            "BROWSER_LAUNCH",
            result="attempted",
            normalized_host=(urllib.parse.urlsplit(final_url).hostname or ""),
        )
        if not self.browser_opener(final_url):
            self._trace("BROWSER_LAUNCH", result="failed")
            raise WebError("o navegador padrao recusou a abertura da URL")
        self._trace("BROWSER_LAUNCH", result="opened")
        return {
            "ok": True,
            "url": final_url,
            "title": inspected.get("title"),
            "browser_opened": True,
            "preflight_status": preflight_status,
            "preflight_limited": preflight_limited,
            "threat_checked": self.threat_analyzer is not None,
            "citation": inspected.get("citation"),
            "notice": (
                "O acesso HTTP automatizado recebeu erro, mas isso não prova "
                "indisponibilidade no navegador; a URL segura foi aberta."
                if preflight_limited
                else None
            ),
        }

    def extract(
        self,
        *,
        url: str,
        query: str,
        max_passages: int = 5,
        passage_chars: int = 1200,
        page_start: int | None = None,
        page_end: int | None = None,
    ) -> dict[str, Any]:
        opened = self.open(
            url=url,
            max_chars=self.config.max_text_chars,
            page_start=page_start,
            page_end=page_end,
        )
        passages = self._rank_passages(
            opened["text"],
            query,
            max_passages=max_passages,
            passage_chars=passage_chars,
        )
        result = {
            "ok": True,
            "url": opened["url"],
            "title": opened["title"],
            "document_type": opened["document_type"],
            "fetched_at": opened["fetched_at"],
            "published_at": opened["published_at"],
            "query": query,
            "passages": passages,
            "notice": "Passagens sao extratos; confira contexto na fonte quando necessario.",
        }
        if "citation" in opened:
            result["citation"] = opened["citation"]
        if "validation" in opened:
            result["validation"] = opened["validation"]
            result["accepted_for_citation"] = opened.get(
                "accepted_for_citation", False
            )
        return result

    def _ensure_enabled(self) -> None:
        if not self.config.enabled:
            raise WebAccessDenied("ferramentas web desativadas por WEB_ENABLED=false")

    def _fetch(
        self,
        raw_url: str,
        *,
        policy: NetworkPolicy | None = None,
        headers: dict[str, str] | None = None,
        operation: str = "open",
    ) -> FetchResponse:
        active_policy = policy or self.policy
        try:
            parsed_input = urllib.parse.urlsplit(raw_url.strip())
            normalized_host = (parsed_input.hostname or "").casefold() or None
            self._trace(
                "URL_NORMALIZATION",
                result="parsed",
                normalized_host=normalized_host,
            )
        except (AttributeError, ValueError):
            self._trace(
                "URL_NORMALIZATION",
                result="blocked",
                reason_code="INVALID_URL",
            )
            raise
        try:
            active_policy.validate_url(raw_url, resolve_dns=False)
        except WebError as exc:
            self._trace(
                "URL_POLICY_CHECK",
                result="blocked",
                reason_code=exc.code,
                normalized_host=normalized_host,
            )
            raise
        self._trace(
            "URL_POLICY_CHECK",
            result="allowed",
            normalized_host=normalized_host,
        )
        try:
            safe_url = active_policy.validate_url(raw_url)
        except WebError as exc:
            self._trace(
                "DNS_IP_VALIDATION",
                result="blocked",
                reason_code=exc.code,
                normalized_host=normalized_host,
            )
            raise
        self._trace(
            "DNS_IP_VALIDATION",
            result="allowed",
            normalized_host=normalized_host,
        )
        self._trace(
            "HTTP_FETCH",
            result="attempted",
            normalized_host=normalized_host,
        )
        if self.transport is not None:
            started = time.monotonic()
            try:
                response = self.transport(safe_url)
            except (socket.timeout, TimeoutError) as exc:
                if operation == "search":
                    raise SearchTimeout(
                        "timeout no provedor de busca",
                        details={
                            "provider": self.config.search_provider,
                            "duration_ms": round(
                                (time.monotonic() - started) * 1000
                            ),
                        },
                    ) from exc
                raise WebTimeout("timeout ao abrir URL") from exc
            except socket.gaierror as exc:
                if operation == "search":
                    raise SearchDnsError(
                        "falha DNS no provedor de busca",
                        details={
                            "provider": self.config.search_provider,
                            "reason": str(exc)[:500],
                        },
                    ) from exc
                raise WebError(f"falha DNS: {exc}") from exc
            except OSError as exc:
                if operation == "search":
                    raise SearchHttpError(
                        "falha de rede no provedor de busca",
                        details={
                            "provider": self.config.search_provider,
                            "reason": str(exc)[:500],
                        },
                    ) from exc
                raise WebError(f"falha de rede: {exc}") from exc
            try:
                final_url = active_policy.validate_url(response.url)
            except WebError as exc:
                self._trace(
                    "REDIRECT_VALIDATION",
                    result="blocked",
                    reason_code=exc.code,
                )
                raise
            self._trace(
                "REDIRECT_VALIDATION",
                result="allowed",
                normalized_host=(
                    urllib.parse.urlsplit(final_url).hostname or ""
                ).casefold(),
            )
            if len(response.data) > self.config.max_download_bytes:
                raise WebTooLarge(
                    f"download excede {self.config.max_download_bytes} bytes"
                )
            normalized = FetchResponse(
                final_url,
                response.status,
                {key.lower(): value for key, value in response.headers.items()},
                response.data,
                response.duration_ms,
            )
            self._validate_search_http(normalized, operation)
            return normalized
        response = self._urllib_fetch(
            safe_url,
            active_policy,
            headers=headers,
            operation=operation,
        )
        self._trace(
            "REDIRECT_VALIDATION",
            result="allowed",
            normalized_host=(
                urllib.parse.urlsplit(response.url).hostname or ""
            ).casefold(),
        )
        return response

    def _urllib_fetch(
        self,
        safe_url: str,
        policy: NetworkPolicy,
        *,
        headers: dict[str, str] | None = None,
        operation: str = "open",
    ) -> FetchResponse:
        request_headers = {
            "User-Agent": self.config.user_agent,
            "Accept": "text/html, application/xhtml+xml, application/pdf, text/plain;q=0.8",
            "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.7",
            "Accept-Encoding": "identity",
            "Connection": "close",
        }
        request_headers.update(headers or {})
        request = urllib.request.Request(
            safe_url,
            method="GET",
            headers=request_headers,
        )
        opener = urllib.request.build_opener(
            _SafeRedirectHandler(policy),
            urllib.request.HTTPSHandler(context=ssl.create_default_context()),
        )
        started = time.monotonic()
        try:
            with opener.open(request, timeout=self.config.timeout) as response:
                final_url = policy.validate_url(response.geturl())
                headers = {
                    key.lower(): value for key, value in response.headers.items()
                }
                length = headers.get("content-length")
                if length and int(length) > self.config.max_download_bytes:
                    raise WebTooLarge(
                        f"download excede {self.config.max_download_bytes} bytes"
                    )
                data = response.read(self.config.max_download_bytes + 1)
                if len(data) > self.config.max_download_bytes:
                    raise WebTooLarge(
                        f"download excede {self.config.max_download_bytes} bytes"
                    )
                result = FetchResponse(
                    final_url,
                    getattr(response, "status", 200),
                    headers,
                    data,
                    round((time.monotonic() - started) * 1000),
                )
                self._validate_search_http(result, operation)
                return result
        except WebError:
            raise
        except urllib.error.HTTPError as exc:
            duration_ms = round((time.monotonic() - started) * 1000)
            data = exc.read(4096)
            details = {
                "provider": self.config.search_provider,
                "status": exc.code,
                "content_type": exc.headers.get("content-type"),
                "duration_ms": duration_ms,
                "body_preview": self._sanitized_preview(data),
            }
            if operation == "browser":
                final_url = policy.validate_url(exc.geturl())
                return FetchResponse(
                    final_url,
                    exc.code,
                    {key.lower(): value for key, value in exc.headers.items()},
                    data,
                    duration_ms,
                )
            if operation == "search" and exc.code in {401, 403}:
                raise SearchAuthenticationFailed(
                    f"autenticacao recusada pelo provedor HTTP {exc.code}",
                    details=details,
                ) from exc
            if operation == "search":
                raise SearchHttpError(
                    f"provedor retornou HTTP {exc.code}",
                    details=details,
                ) from exc
            details["availability_unknown"] = True
            raise WebError(
                f"acesso HTTP automatizado recebeu {exc.code}; "
                "isso não comprova indisponibilidade no navegador",
                details=details,
            ) from exc
        except urllib.error.URLError as exc:
            duration_ms = round((time.monotonic() - started) * 1000)
            details = {
                "provider": self.config.search_provider,
                "duration_ms": duration_ms,
                "reason": str(exc.reason)[:500],
            }
            if operation == "search" and isinstance(
                exc.reason, socket.gaierror
            ):
                raise SearchDnsError(
                    "falha DNS no provedor de busca", details=details
                ) from exc
            if operation == "search":
                raise SearchHttpError(
                    "falha de rede no provedor de busca", details=details
                ) from exc
            raise WebError(
                f"falha de rede ao abrir {safe_url}: {exc}", details=details
            ) from exc
        except (socket.timeout, TimeoutError) as exc:
            if time.monotonic() - started >= self.config.timeout:
                if operation == "search":
                    raise SearchTimeout(
                        "timeout no provedor de busca",
                        details={
                            "provider": self.config.search_provider,
                            "duration_ms": round(
                                (time.monotonic() - started) * 1000
                            ),
                        },
                    ) from exc
                raise WebTimeout(f"timeout ao abrir {safe_url}") from exc
            raise

    def _validate_search_http(
        self, response: FetchResponse, operation: str
    ) -> None:
        if operation != "search" or 200 <= response.status < 300:
            return
        details = {
            "provider": self.config.search_provider,
            "status": response.status,
            "content_type": response.headers.get("content-type"),
            "duration_ms": response.duration_ms,
            "body_preview": self._sanitized_preview(response.data),
        }
        if response.status in {401, 403}:
            raise SearchAuthenticationFailed(
                f"autenticacao recusada pelo provedor HTTP {response.status}",
                details=details,
            )
        raise SearchHttpError(
            f"provedor retornou HTTP {response.status}",
            details=details,
        )

    @staticmethod
    def _decode(response: FetchResponse) -> str:
        content_type = response.headers.get("content-type", "")
        match = re.search(r"charset=([^\s;]+)", content_type, re.IGNORECASE)
        charset = match.group(1).strip("\"'") if match else "utf-8"
        try:
            return response.data.decode(charset, errors="replace")
        except LookupError:
            return response.data.decode("utf-8", errors="replace")

    def _extract_html(self, response: FetchResponse) -> dict[str, Any]:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise WebError(
                "BeautifulSoup ausente; instale beautifulsoup4 para leitura HTML"
            ) from exc
        soup = BeautifulSoup(self._decode(response), "html.parser")
        for element in soup(
            ["script", "style", "noscript", "svg", "form", "nav", "footer", "aside"]
        ):
            element.decompose()
        title_node = soup.find("meta", property="og:title") or soup.find("title")
        if getattr(title_node, "name", None) == "meta":
            title = title_node.get("content", "")
        else:
            title = (
                title_node.get_text(" ", strip=True) if title_node else response.url
            )
        published_at = None
        for selector in (
            {"property": "article:published_time"},
            {"name": "date"},
            {"name": "pubdate"},
            {"name": "publish-date"},
        ):
            node = soup.find("meta", attrs=selector)
            if node and node.get("content"):
                published_at = node["content"].strip()
                break
        if published_at is None:
            time_node = soup.find("time", attrs={"datetime": True})
            if time_node:
                published_at = time_node.get("datetime")
        container = soup.find("article") or soup.find("main") or soup.body or soup
        blocks = []
        for node in container.find_all(
            ["h1", "h2", "h3", "h4", "p", "li", "blockquote", "pre", "table"]
        ):
            text = _normalize_space(node.get_text(" ", strip=True))
            if text and (not blocks or blocks[-1] != text):
                blocks.append(text)
        if not blocks:
            blocks = [_normalize_space(container.get_text("\n", strip=True))]
        links = []
        seen = set()
        for node in container.find_all("a", href=True):
            absolute = urllib.parse.urljoin(response.url, node["href"])
            try:
                safe_url = self.policy.validate_url(absolute, resolve_dns=False)
            except WebError:
                continue
            if safe_url in seen:
                continue
            seen.add(safe_url)
            links.append(
                {
                    "text": _normalize_space(node.get_text(" ", strip=True))[:200],
                    "url": safe_url,
                }
            )
            if len(links) >= 40:
                break
        return {
            "title": _normalize_space(title) or response.url,
            "text": "\n\n".join(blocks),
            "links": links,
            "published_at": published_at,
            "document_type": "html",
        }

    def _extract_pdf(
        self,
        response: FetchResponse,
        page_start: int | None,
        page_end: int | None,
    ) -> dict[str, Any]:
        try:
            import pymupdf
        except ImportError as exc:
            raise WebError("PyMuPDF ausente; instale pymupdf para leitura PDF") from exc
        try:
            document = pymupdf.open(stream=response.data, filetype="pdf")
        except Exception as exc:
            raise WebError(f"PDF invalido: {exc}") from exc
        try:
            if document.needs_pass:
                raise WebError("PDF protegido por senha")
            total_pages = document.page_count
            start = page_start or 1
            if start > max(total_pages, 1):
                raise WebError(
                    f"page_start excede total de {total_pages} paginas"
                )
            requested_end = page_end or total_pages
            end = min(
                requested_end,
                total_pages,
                start + self.config.max_pdf_pages - 1,
            )
            blocks = []
            links = []
            seen = set()
            for page_number in range(start - 1, end):
                page = document.load_page(page_number)
                text = _normalize_space(page.get_text("text"))
                if text:
                    blocks.append(f"[Pagina {page_number + 1}]\n{text}")
                for link in page.get_links():
                    uri = link.get("uri")
                    if not uri:
                        continue
                    try:
                        safe_url = self.policy.validate_url(
                            uri, resolve_dns=False
                        )
                    except WebError:
                        continue
                    if safe_url not in seen:
                        seen.add(safe_url)
                        links.append({"text": "", "url": safe_url})
            metadata = document.metadata or {}
            return {
                "title": _normalize_space(metadata.get("title") or "")
                or urllib.parse.urlsplit(response.url).path.rsplit("/", 1)[-1]
                or response.url,
                "text": "\n\n".join(blocks),
                "links": links[:40],
                "published_at": metadata.get("creationDate") or None,
                "document_type": "pdf",
                "page_count": total_pages,
                "pages_read": {"start": start, "end": end},
                "pages_truncated": end < requested_end or end < total_pages,
            }
        finally:
            document.close()

    @staticmethod
    def _parse_search_results(html: str) -> list[tuple[str, str, str]]:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise WebError(
                "BeautifulSoup ausente; instale beautifulsoup4 para pesquisa web"
            ) from exc
        soup = BeautifulSoup(html, "html.parser")
        results = []
        for item in soup.select(".result"):
            anchor = item.select_one("a.result__a")
            if anchor is None or not anchor.get("href"):
                continue
            url = WebClient._unwrap_search_url(anchor["href"])
            snippet_node = item.select_one(".result__snippet")
            results.append(
                (
                    _normalize_space(anchor.get_text(" ", strip=True)),
                    url,
                    _normalize_space(
                        snippet_node.get_text(" ", strip=True)
                        if snippet_node
                        else ""
                    ),
                )
            )
        return results

    @staticmethod
    def _unwrap_search_url(raw_url: str) -> str:
        absolute = urllib.parse.urljoin("https://duckduckgo.com/", raw_url)
        parsed = urllib.parse.urlsplit(absolute)
        query = urllib.parse.parse_qs(parsed.query)
        if (
            parsed.hostname
            and parsed.hostname.endswith("duckduckgo.com")
            and query.get("uddg")
        ):
            return query["uddg"][0]
        return absolute

    @staticmethod
    def _duckduckgo_region(language: str) -> str:
        normalized = language.lower()
        if normalized == "pt-br":
            return "br-pt"
        if normalized == "pt-pt":
            return "pt-pt"
        return normalized

    @staticmethod
    def _freshness_bucket(days: int | None) -> str | None:
        if days is None:
            return None
        if days <= 1:
            return "d"
        if days <= 7:
            return "w"
        if days <= 31:
            return "m"
        return "y"

    @staticmethod
    def _rank_passages(
        text: str,
        query: str,
        *,
        max_passages: int,
        passage_chars: int,
    ) -> list[dict[str, Any]]:
        query_tokens = {
            token.lower()
            for token in _TOKEN_RE.findall(query)
            if token.lower() not in _STOPWORDS
        }
        raw_blocks = [
            block.strip()
            for block in re.split(r"\n{2,}", text)
            if block.strip()
        ]
        blocks = []
        for block in raw_blocks:
            if len(block) <= passage_chars:
                blocks.append(block)
            else:
                for offset in range(0, len(block), passage_chars):
                    blocks.append(block[offset : offset + passage_chars])
        scored = []
        query_lower = query.lower().strip()
        for index, block in enumerate(blocks):
            lower = block.lower()
            tokens = _TOKEN_RE.findall(lower)
            overlap = sum(tokens.count(token) for token in query_tokens)
            coverage = sum(1 for token in query_tokens if token in lower)
            exact = 3 if len(query_lower) >= 4 and query_lower in lower else 0
            score = overlap + coverage * 2 + exact
            if score:
                scored.append((score, index, block))
        if not scored:
            scored = [
                (0, index, block)
                for index, block in enumerate(blocks[:max_passages])
            ]
        selected = sorted(scored, key=lambda item: (-item[0], item[1]))[
            :max_passages
        ]
        return [
            {"index": index + 1, "score": score, "text": block}
            for score, index, block in selected
        ]
