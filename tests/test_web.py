from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from tern.orchestrator.agent import Supervisor
from tern.orchestrator.config import load_settings
from tern.orchestrator.security import ActionLogger, PathPolicy
from tern.orchestrator.tools import ToolRegistry
from tern.orchestrator.web import (
    FetchResponse,
    NetworkPolicy,
    WebAccessDenied,
    WebClient,
    WebConfig,
    SearchAuthenticationFailed,
    SearchHttpError,
    SearchProviderNotConfigured,
    SearchResponseInvalid,
    SearchTimeout,
    WebError,
    WebTooLarge,
    _open_system_browser,
)


PUBLIC_IP = "93.184.216.34"


@pytest.fixture(autouse=True)
def prevent_unmocked_real_browser_launch(monkeypatch):
    monkeypatch.setattr(
        "tern.orchestrator.web._open_system_browser",
        lambda url: pytest.fail(f"test attempted real browser launch: {url}"),
    )


def public_resolver(_host: str):
    return [PUBLIC_IP]


def response(url: str, content_type: str, data: bytes) -> FetchResponse:
    return FetchResponse(url, 200, {"content-type": content_type}, data)


class FakeCodex:
    timeout = 1

    def delegate(self, _task):
        raise AssertionError("not expected")

    def continue_session(self, **_arguments):
        raise AssertionError("not expected")


class FakeModel:
    def __init__(self, values):
        self.values = iter(values)

    def chat(self, _messages, **_kwargs):
        return next(self.values)


def web_registry(tmp_path: Path, web: WebClient) -> ToolRegistry:
    return ToolRegistry(
        policy=PathPolicy((tmp_path,)),
        logger=ActionLogger(tmp_path / "actions.jsonl"),
        codex=FakeCodex(),
        max_output_bytes=131072,
        web=web,
    )


def test_network_policy_blocks_ssrf_and_unsafe_urls():
    policy = NetworkPolicy(resolver=public_resolver)
    with pytest.raises(WebAccessDenied):
        policy.validate_url("http://localhost/admin")
    with pytest.raises(WebAccessDenied):
        policy.validate_url("http://127.0.0.1/admin")
    with pytest.raises(WebAccessDenied):
        policy.validate_url("http://10.0.0.10/admin")
    with pytest.raises(WebAccessDenied):
        policy.validate_url("http://169.254.169.254/latest/meta-data/")
    with pytest.raises(WebAccessDenied):
        policy.validate_url("file:///etc/passwd")
    with pytest.raises(WebAccessDenied):
        policy.validate_url("https://user:password@example.com/")
    with pytest.raises(WebAccessDenied):
        policy.validate_url("https://example.com:8443/")

    rebinding = NetworkPolicy(resolver=lambda _host: ["10.0.0.10"])
    with pytest.raises(WebAccessDenied):
        rebinding.validate_url("https://example.com/")


def test_safe_search_off_does_not_relax_network_policy():
    client = WebClient(
        WebConfig(safe_search="off"),
        resolver=lambda _host: ["10.0.0.10"],
        transport=lambda _url: pytest.fail("unsafe URL must not be fetched"),
    )
    with pytest.raises(WebAccessDenied):
        client.open(url="https://malicious.example/payload")


def test_redirect_to_private_destination_remains_blocked():
    client = WebClient(
        WebConfig(),
        resolver=public_resolver,
        transport=lambda _url: response(
            "http://127.0.0.1/private", "text/html", b"<html></html>"
        ),
    )
    events = []
    client.set_trace_callback(
        lambda stage, values: events.append({"stage": stage, **values})
    )
    with pytest.raises(WebAccessDenied):
        client.open(url="https://public.example/start")
    assert any(
        event["stage"] == "REDIRECT_VALIDATION"
        and event["result"] == "blocked"
        for event in events
    )


def test_browser_launch_runs_only_after_safe_fetch_and_threat_analysis(tmp_path):
    opened = []
    events = []
    client = WebClient(
        WebConfig(threat_memory_path=tmp_path / "patterns.json"),
        resolver=public_resolver,
        transport=lambda url: response(
            "https://www.example.com/final",
            "text/html",
            b"<html><head><title>Safe</title></head><body>ok</body></html>",
        ),
        browser_opener=lambda url: opened.append(url) or True,
    )
    client.set_trace_callback(
        lambda stage, values: events.append({"stage": stage, **values})
    )
    result = client.open_in_browser(url="https://example.com/start")
    assert result["browser_opened"] is True
    assert opened == ["https://www.example.com/final"]
    stages = [item["stage"] for item in events]
    assert stages.index("DNS_IP_VALIDATION") < stages.index("HTTP_FETCH")
    assert stages.index("WEB_THREAT_ANALYSIS") < stages.index("BROWSER_LAUNCH")


def test_system_browser_opener_uses_new_tab_in_existing_session(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "tern.orchestrator.web._windows_running_browser_executable",
        lambda: r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
    )
    monkeypatch.setattr(
        "tern.orchestrator.web._windows_default_browser_executable",
        lambda: r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    )
    monkeypatch.setattr(
        "tern.orchestrator.web.subprocess.Popen",
        lambda arguments, **options: calls.append((arguments, options)),
    )

    assert _open_system_browser("https://www.amazon.com.br/") is True
    assert calls[0][0] == [
        r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
        "https://www.amazon.com.br/",
    ]
    assert "--new-window" not in calls[0][0]


def test_system_browser_opener_uses_default_when_no_browser_window_exists(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "tern.orchestrator.web._windows_running_browser_executable",
        lambda: None,
    )
    monkeypatch.setattr(
        "tern.orchestrator.web._windows_default_browser_executable",
        lambda: r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    )
    monkeypatch.setattr(
        "tern.orchestrator.web.subprocess.Popen",
        lambda arguments, **options: calls.append((arguments, options)),
    )

    assert _open_system_browser("https://www.gov.br/pt-br") is True
    assert calls[0][0] == [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        "https://www.gov.br/pt-br",
    ]


def test_browser_is_not_launched_when_threat_analysis_blocks(tmp_path):
    opened = []
    client = WebClient(
        WebConfig(threat_memory_path=tmp_path / "patterns.json"),
        resolver=public_resolver,
        transport=lambda url: response(
            url,
            "text/html",
            b'<html><iframe src="http://127.0.0.1/admin"></iframe></html>',
        ),
        browser_opener=lambda url: opened.append(url) or True,
    )
    with pytest.raises(WebAccessDenied):
        client.open_in_browser(url="https://malicious.example/")
    assert opened == []


def test_browser_launches_when_automated_preflight_returns_503(tmp_path):
    opened = []
    events = []
    client = WebClient(
        WebConfig(threat_memory_path=tmp_path / "patterns.json"),
        resolver=public_resolver,
        transport=lambda url: FetchResponse(
            url,
            503,
            {"content-type": "text/html"},
            b"<html><head><title>Robot blocked</title></head><body>unavailable</body></html>",
        ),
        browser_opener=lambda url: opened.append(url) or True,
    )
    client.set_trace_callback(
        lambda stage, values: events.append({"stage": stage, **values})
    )

    result = client.open_in_browser(url="https://www.amazon.com.br/")

    assert result["browser_opened"] is True
    assert result["preflight_status"] == 503
    assert result["preflight_limited"] is True
    assert result["citation"] is None
    assert "não prova indisponibilidade" in result["notice"]
    assert opened == ["https://www.amazon.com.br/"]
    assert any(
        event["stage"] == "BROWSER_PREFLIGHT"
        and event["reason_code"] == "HTTP_503"
        for event in events
    )


@pytest.mark.parametrize(
    ("user_text", "expected_url"),
    [
        (
            "bota pra tocar a musica trust bothbirds",
            "https://www.youtube.com/results?search_query=trust+bothbirds",
        ),
        (
            "toque trust bothbirds no spotify",
            "https://open.spotify.com/search/trust%20bothbirds",
        ),
    ],
)
def test_music_request_runs_security_checks_then_opens_requested_platform(
    tmp_path,
    user_text,
    expected_url,
):
    opened = []
    fetched = []

    def transport(url):
        fetched.append(url)
        return response(
            url,
            "text/html",
            b"<html><head><title>Music</title></head><body>safe</body></html>",
        )

    result = Supervisor(
        load_settings({"MODEL_MAX_TOOL_CALLS": "1"}),
        FakeModel(
            [
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Música aberta.",
                            }
                        }
                    ]
                }
            ]
        ),
        web_registry(
            tmp_path,
            WebClient(
                WebConfig(threat_memory_path=tmp_path / "patterns.json"),
                resolver=public_resolver,
                transport=transport,
                browser_opener=lambda url: opened.append(url) or True,
            ),
        ),
    ).run(user_text)

    assert result["decision"]["reason_code"] == "music_browser_search"
    assert fetched == [expected_url]
    assert opened == [expected_url]
    stages = [item["stage"] for item in result["pipeline_trace"]]
    assert stages.index("DNS_IP_VALIDATION") < stages.index("HTTP_FETCH")
    assert stages.index("WEB_THREAT_ANALYSIS") < stages.index("BROWSER_LAUNCH")


def test_browser_followup_opens_last_validated_url_in_one_tool_call(tmp_path):
    opened = []
    html = b"<html><head><title>Opened</title></head><body>safe</body></html>"
    model = FakeModel(
        [
            {"choices": [{"message": {"role": "assistant", "content": "Pagina lida."}}]},
            {"choices": [{"message": {"role": "assistant", "content": "Guia aberta."}}]},
        ]
    )
    supervisor = Supervisor(
        load_settings({"MODEL_MAX_TOOL_CALLS": "1"}),
        model,
        web_registry(
            tmp_path,
            WebClient(
                WebConfig(threat_memory_path=tmp_path / "patterns.json"),
                resolver=public_resolver,
                transport=lambda url: response(url, "text/html", html),
                browser_opener=lambda url: opened.append(url) or True,
            ),
        ),
    )
    first = supervisor.run("abra https://example.com")
    second = supervisor.run("abre ai no meu navegador cara")
    assert first["ok"] is True
    assert first["decision"]["reason_code"] == "explicit_browser_url"
    assert second["decision"]["reason_code"] == "explicit_browser_url"
    assert second["tool_calls"] == 1
    assert opened == ["https://example.com/", "https://example.com/"]


@pytest.mark.parametrize(
    ("prompt", "expected_url"),
    [
        ("abra xvideos", "https://www.xvideos.com/"),
        ("abra o pornhub", "https://pt.pornhub.com/"),
        ("abra a amazon", "https://www.amazon.com.br/"),
    ],
)
def test_known_site_alias_opens_browser_without_url(tmp_path, prompt, expected_url):
    opened = []
    model = FakeModel(
        [{"choices": [{"message": {"role": "assistant", "content": "Site aberto."}}]}]
    )
    supervisor = Supervisor(
        load_settings({"MODEL_MAX_TOOL_CALLS": "1"}),
        model,
        web_registry(
            tmp_path,
            WebClient(
                WebConfig(threat_memory_path=tmp_path / "patterns.json"),
                resolver=public_resolver,
                transport=lambda url: response(
                    url,
                    "text/html",
                    b"<html><head><title>XVideos</title></head><body>ok</body></html>",
                ),
                browser_opener=lambda url: opened.append(url) or True,
            ),
        ),
    )

    result = supervisor.run(prompt)

    assert result["ok"] is True
    assert result["decision"]["reason_code"] == "known_site_browser_url"
    assert result["tool_calls"] == 1
    assert opened == [expected_url]


@pytest.mark.parametrize("host", ["example.com", "xvideos.com"])
def test_explicit_public_hostname_reaches_web_pipeline(tmp_path, host):
    calls = []
    opened = []
    html = b"<html><head><title>Opened</title></head><body>safe</body></html>"

    def transport(url: str):
        calls.append(url)
        return response(url, "text/html", html)

    model = FakeModel(
        [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Pagina aberta.",
                        }
                    }
                ]
            }
        ]
    )
    result = Supervisor(
        load_settings({"MODEL_MAX_TOOL_CALLS": "1"}),
        model,
        web_registry(
            tmp_path,
            WebClient(
                WebConfig(threat_memory_path=tmp_path / "patterns.json"),
                resolver=public_resolver,
                transport=transport,
                browser_opener=lambda url: opened.append(url) or True,
            ),
        ),
    ).run(f"abra {host}")
    assert result["decision"]["intent"] == "WEB_OPEN"
    assert result["tool_calls"] == 1
    assert result["web"]["used"]
    assert urlsplit(calls[0]).hostname == host
    assert urlsplit(opened[0]).hostname == host
    stages = [item["stage"] for item in result["pipeline_trace"]]
    assert "HTTP_FETCH" in stages
    assert "WEB_THREAT_ANALYSIS" in stages
    assert "CONTENT_EXTRACTION" in stages


@pytest.mark.parametrize(
    "url_request",
    [
        "abra http://localhost/admin",
        "abra http://127.0.0.1/admin",
        "abra http://10.0.0.10/admin",
        "abra http://169.254.169.254/latest/meta-data/",
        "abra file:///etc/passwd",
    ],
)
def test_explicit_unsafe_url_reaches_policy_but_never_transport(
    tmp_path, url_request
):
    model = FakeModel(
        [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Destino bloqueado com seguranca.",
                        }
                    }
                ]
            }
        ]
    )
    web = WebClient(
        WebConfig(threat_memory_path=tmp_path / "patterns.json"),
        resolver=public_resolver,
        transport=lambda _url: pytest.fail("unsafe destination was fetched"),
    )
    result = Supervisor(
        load_settings({"MODEL_MAX_TOOL_CALLS": "1"}),
        model,
        web_registry(tmp_path, web),
    ).run(url_request)
    assert result["decision"]["intent"] == "WEB_OPEN"
    assert result["tool_calls"] == 1
    assert result["web"]["used"]
    policy = next(
        item
        for item in result["pipeline_trace"]
        if item["stage"] == "URL_POLICY_CHECK"
    )
    assert policy["result"] == "blocked"
    assert result["answer"].startswith("Não abri o site.")
    assert "Relatório de segurança:" in result["answer"]
    assert "Guia aberta: não." in result["answer"]
    assert any(
        item["stage"] == "WEB_SAFETY_REPORT"
        for item in result["pipeline_trace"]
    )


def test_threat_block_returns_specific_report_and_never_opens_browser(tmp_path):
    opened = []
    client = WebClient(
        WebConfig(threat_memory_path=tmp_path / "patterns.json"),
        resolver=public_resolver,
        transport=lambda url: response(
            url,
            "text/html",
            b'<html><script>fetch("http://169.254.169.254/data")</script></html>',
        ),
        browser_opener=lambda url: opened.append(url) or True,
    )
    result = Supervisor(
        load_settings({"MODEL_MAX_TOOL_CALLS": "1"}),
        FakeModel([]),
        web_registry(tmp_path, client),
    ).run("abra https://malicious.example")

    assert result["answer"].startswith("Não abri o site.")
    assert "rede local/reservada" in result["answer"]
    assert "Guia aberta: não." in result["answer"]
    assert opened == []


def test_pipeline_trace_contains_only_technical_fields(tmp_path):
    secret_page_text = "PAGE_CONTENT_MUST_NOT_ENTER_TRACE"
    model = FakeModel(
        [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Concluido.",
                        }
                    }
                ]
            }
        ]
    )
    web = WebClient(
        WebConfig(threat_memory_path=tmp_path / "patterns.json"),
        resolver=public_resolver,
        transport=lambda url: response(
            url,
            "text/html",
            f"<html><body>{secret_page_text}</body></html>".encode(),
        ),
        browser_opener=lambda _url: True,
    )
    result = Supervisor(
        load_settings({"MODEL_MAX_TOOL_CALLS": "1"}),
        model,
        web_registry(tmp_path, web),
    ).run("abra https://example.com")
    serialized = json.dumps(result["pipeline_trace"])
    assert secret_page_text not in serialized
    allowed = {
        "stage",
        "result",
        "reason_code",
        "tool_name",
        "normalized_host",
        "decision_type",
        "registered",
        "exposed",
        "web_enabled",
        "ambiguity_present",
        "requested_agent",
        "requested_agent_source",
        "tool_available",
        "execution_allowed",
    }
    assert all(set(item) <= allowed for item in result["pipeline_trace"])


def test_active_private_reference_is_blocked_and_learned(tmp_path):
    memory = tmp_path / "web-threat-patterns.json"
    calls = []
    dangerous = b"""
    <html><body>
      <img src="http://127.0.0.1:80/admin">
    </body></html>
    """

    def transport(url: str):
        calls.append(url)
        return response(url, "text/html", dangerous)

    client = WebClient(
        WebConfig(threat_memory_path=memory),
        resolver=public_resolver,
        transport=transport,
    )
    with pytest.raises(WebAccessDenied) as first:
        client.open(url="https://malicious.example/page")
    assert first.value.details["codes"] == [
        "ACTIVE_LOCAL_NETWORK_REFERENCE"
    ]
    assert not first.value.details["learned_match"]
    assert memory.is_file()

    with pytest.raises(WebAccessDenied) as second:
        client.open(url="https://malicious.example/another")
    assert second.value.details["codes"] == ["KNOWN_MALICIOUS_HOST"]
    assert second.value.details["learned_match"]
    assert len(calls) == 1


def test_learned_pattern_is_recognized_across_domains(tmp_path):
    memory = tmp_path / "web-threat-patterns.json"
    dangerous = b'<html><body><iframe src="http://10.0.0.2/x"></iframe></body></html>'
    client = WebClient(
        WebConfig(threat_memory_path=memory),
        resolver=public_resolver,
        transport=lambda url: response(url, "text/html", dangerous),
    )
    with pytest.raises(WebAccessDenied) as first:
        client.open(url="https://first.example/")
    with pytest.raises(WebAccessDenied) as second:
        client.open(url="https://second.example/")
    assert first.value.details["pattern_id"] == second.value.details["pattern_id"]
    assert not first.value.details["learned_match"]
    assert second.value.details["learned_match"]


def test_script_request_to_local_machine_is_blocked(tmp_path):
    body = b"""
    <html><body><script>
      fetch('http://169.254.169.254/latest/meta-data/');
    </script></body></html>
    """
    client = WebClient(
        WebConfig(threat_memory_path=tmp_path / "patterns.json"),
        resolver=public_resolver,
        transport=lambda url: response(url, "text/html", body),
    )
    with pytest.raises(WebAccessDenied) as captured:
        client.open(url="https://malicious.example/")
    assert captured.value.details["signals"][0]["context"] == (
        "script.network-sink"
    )
    assert captured.value.details["signals"][0]["target_class"] == (
        "link_local_or_metadata"
    )


def test_normal_links_and_public_assets_do_not_poison_memory(tmp_path):
    body = b"""
    <html><head><title>Safe</title></head><body>
      <a href="http://127.0.0.1/manual-example">textual example</a>
      <img src="https://cdn.example/image.png">
      <script>window.note = 'do not trust this site';</script>
      <p>Safe content.</p>
    </body></html>
    """
    memory = tmp_path / "patterns.json"
    client = WebClient(
        WebConfig(threat_memory_path=memory),
        resolver=public_resolver,
        transport=lambda url: response(url, "text/html", body),
    )
    opened = client.open(url="https://safe.example/")
    assert opened["ok"]
    assert opened["links"] == []
    assert not memory.exists()
    assert client.research_status()["threat_analysis"] == {
        "learning_enabled": True,
        "known_malicious_hosts": 0,
        "learned_patterns": 0,
    }


@pytest.mark.parametrize(
    ("provider", "search_url", "expected_key", "expected_value"),
    [
        ("bing_rss", "https://www.bing.com/search", "adlt", "off"),
        (
            "brave",
            "https://api.search.brave.com/res/v1/web/search",
            "safesearch",
            "off",
        ),
        (
            "duckduckgo_html",
            "https://html.duckduckgo.com/html/",
            "kp",
            "-2",
        ),
    ],
)
def test_safe_search_off_is_forwarded_to_each_provider(
    provider, search_url, expected_key, expected_value
):
    calls = []

    def transport(url: str):
        calls.append(url)
        if provider == "bing_rss":
            return response(url, "text/xml", b"<rss><channel></channel></rss>")
        if provider == "brave":
            return response(url, "application/json", b'{"web":{"results":[]}}')
        return response(
            url,
            "text/html",
            b'<html><body><div class="result__a"></div></body></html>',
        )

    client = WebClient(
        WebConfig(
            search_provider=provider,
            search_url=search_url,
            search_api_key="test-key" if provider == "brave" else None,
            safe_search="off",
            max_research_corrections=0,
        ),
        resolver=public_resolver,
        transport=transport,
    )
    client.search(query="adult content", language="en")
    parameters = parse_qs(urlsplit(calls[0]).query)
    assert parameters[expected_key] == [expected_value]


def test_search_normalizes_results_and_filters_domains():
    html = b"""
    <div class="result">
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fnews">Fonte A</a>
      <div class="result__snippet">Trecho A</div>
    </div>
    <div class="result">
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fblocked.test%2Fx">Fonte B</a>
      <div class="result__snippet">Trecho B</div>
    </div>
    """

    def transport(url: str):
        assert urlsplit(url).hostname == "html.duckduckgo.com"
        return response(url, "text/html; charset=utf-8", html)

    client = WebClient(
        WebConfig(
            search_provider="duckduckgo_html",
            search_url="https://html.duckduckgo.com/html/",
        ),
        resolver=public_resolver,
        transport=transport,
    )
    result = client.search(
        query="pesquisa teste",
        allowed_domains=["example.com"],
        blocked_domains=[],
    )
    assert result["ok"]
    assert result["result_count"] == 1
    assert result["results"][0]["url"] == "https://example.com/news"
    assert result["results"][0]["snippet"] == "Trecho A"
    assert "abra fontes" in result["notice"]


def test_bing_rss_real_shape_is_normalized():
    rss = b"""<?xml version="1.0" encoding="utf-8"?>
    <rss version="2.0"><channel><title>Bing</title>
      <item>
        <title>Official documentation</title>
        <link>https://example.com/docs</link>
        <description>&lt;b&gt;Structured&lt;/b&gt; result snippet.</description>
        <pubDate>Wed, 29 Jul 2026 12:00:00 GMT</pubDate>
      </item>
    </channel></rss>"""
    client = WebClient(
        WebConfig(),
        resolver=public_resolver,
        transport=lambda url: FetchResponse(
            url,
            200,
            {"content-type": "text/xml; charset=utf-8"},
            rss,
            17,
        ),
    )
    result = client.search(query="official docs", language="en")
    assert result["provider"] == "bing_rss"
    assert result["http"]["status"] == 200
    assert result["http"]["duration_ms"] == 17
    item = result["results"][0]
    assert item["id"] == 1
    assert item["title"] == "Official documentation"
    assert item["url"] == "https://example.com/docs"
    assert item["domain"] == "example.com"
    assert item["snippet"] == "Structured result snippet."
    assert item["published_at"] == "2026-07-29T12:00:00+00:00"
    assert item["source_type"] == "documentation"
    assert item["final_score"] >= 0.55


def test_duckduckgo_challenge_is_not_silent_zero_results():
    challenge = b"""<html><body>
    <form id="challenge-form" action="/anomaly.js"></form>
    Unfortunately, bots use DuckDuckGo too
    </body></html>"""
    client = WebClient(
        WebConfig(
            search_provider="duckduckgo_html",
            search_url="https://html.duckduckgo.com/html/",
        ),
        resolver=public_resolver,
        transport=lambda url: FetchResponse(
            url,
            202,
            {"content-type": "text/html; charset=UTF-8"},
            challenge,
            35,
        ),
    )
    with pytest.raises(SearchResponseInvalid) as captured:
        client.search(query="test")
    assert captured.value.code == "search_response_invalid"
    assert captured.value.details["status"] == 202
    assert "challenge-form" in captured.value.details["body_preview"]


@pytest.mark.parametrize(
    ("status", "error_type", "code"),
    [
        (401, SearchAuthenticationFailed, "search_authentication_failed"),
        (500, SearchHttpError, "search_http_error"),
    ],
)
def test_search_http_errors_are_structured(status, error_type, code):
    client = WebClient(
        WebConfig(),
        resolver=public_resolver,
        transport=lambda url: FetchResponse(
            url,
            status,
            {"content-type": "text/plain"},
            b"provider error",
            9,
        ),
    )
    with pytest.raises(error_type) as captured:
        client.search(query="test")
    assert captured.value.code == code
    assert captured.value.details["status"] == status
    assert captured.value.details["body_preview"] == "provider error"


def test_invalid_search_response_is_structured():
    client = WebClient(
        WebConfig(),
        resolver=public_resolver,
        transport=lambda url: response(url, "text/xml", b"<not-rss/>"),
    )
    with pytest.raises(SearchResponseInvalid) as captured:
        client.search(query="test")
    assert captured.value.code == "search_response_invalid"


def test_brave_requires_key_before_network():
    client = WebClient(
        WebConfig(
            search_provider="brave",
            search_url="https://api.search.brave.com/res/v1/web/search",
        ),
        resolver=public_resolver,
        transport=lambda _url: pytest.fail("network must not run"),
    )
    with pytest.raises(SearchProviderNotConfigured) as captured:
        client.search(query="test")
    assert captured.value.code == "search_provider_not_configured"


def test_search_timeout_is_structured():
    def timeout(_url):
        raise socket.timeout("late")

    client = WebClient(
        WebConfig(),
        resolver=public_resolver,
        transport=timeout,
    )
    with pytest.raises(SearchTimeout) as captured:
        client.search(query="test")
    assert captured.value.code == "search_timeout"


def test_open_html_extracts_content_metadata_links_and_citation():
    html = b"""
    <html><head><title>Documento teste</title>
    <meta property="article:published_time" content="2026-07-29"></head>
    <body><nav>menu</nav><article>
      <h1>Titulo</h1><p>Informacao verificavel.</p>
      <script>ignore()</script>
      <a href="/more">Mais</a>
      <a href="http://127.0.0.1/private">Privado</a>
    </article></body></html>
    """

    def transport(url: str):
        return response(url, "text/html; charset=utf-8", html)

    client = WebClient(
        WebConfig(
            search_provider="duckduckgo_html",
            search_url="https://html.duckduckgo.com/html/",
        ),
        resolver=public_resolver,
        transport=transport,
    )
    result = client.open(url="https://example.com/article", max_chars=4096)
    assert result["ok"] and result["document_type"] == "html"
    assert result["title"] == "Documento teste"
    assert result["published_at"] == "2026-07-29"
    assert "Informacao verificavel" in result["text"]
    assert "ignore()" not in result["text"]
    assert result["links"] == [
        {"text": "Mais", "url": "https://example.com/more"}
    ]
    assert result["citation"]["url"] == "https://example.com/article"
    assert len(result["sha256"]) == 64


def test_extract_ranks_relevant_passages():
    html = b"""
    <html><head><title>Pesquisa</title></head><body><article>
      <p>Conteudo geral sem relacao direta.</p>
      <p>Qwen3.5 executa chamadas estruturadas de ferramentas JSON.</p>
      <p>Outro assunto.</p>
    </article></body></html>
    """
    client = WebClient(
        WebConfig(),
        resolver=public_resolver,
        transport=lambda url: response(url, "text/html", html),
    )
    result = client.extract(
        url="https://example.com/research",
        query="Qwen3.5 ferramentas JSON",
        max_passages=1,
    )
    assert result["passages"]
    assert "Qwen3.5" in result["passages"][0]["text"]
    assert result["citation"]["title"] == "Pesquisa"


def test_open_pdf_reads_selected_pages():
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    first = document.new_page()
    first.insert_text((72, 72), "Primeira pagina")
    second = document.new_page()
    second.insert_text((72, 72), "Segunda pagina relevante")
    data = document.tobytes()
    document.close()

    client = WebClient(
        WebConfig(max_pdf_pages=1),
        resolver=public_resolver,
        transport=lambda url: response(url, "application/pdf", data),
    )
    result = client.open(
        url="https://example.com/report.pdf",
        page_start=2,
        page_end=2,
    )
    assert result["document_type"] == "pdf"
    assert result["page_count"] == 2
    assert result["pages_read"] == {"start": 2, "end": 2}
    assert "Segunda pagina relevante" in result["text"]
    assert "Primeira pagina" not in result["text"]


def test_download_size_limit_is_enforced():
    client = WebClient(
        WebConfig(max_download_bytes=1024),
        resolver=public_resolver,
        transport=lambda url: response(url, "text/plain", b"x" * 1025),
    )
    with pytest.raises(WebTooLarge):
        client.open(url="https://example.com/large")


def test_web_tools_validate_arguments_and_are_logged(tmp_path):
    search_html = b"""
    <div class="result"><a class="result__a"
    href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2F">Fonte</a></div>
    """
    client = WebClient(
        WebConfig(
            search_provider="duckduckgo_html",
            search_url="https://html.duckduckgo.com/html/",
        ),
        resolver=public_resolver,
        transport=lambda url: response(url, "text/html", search_html),
    )
    tools = web_registry(tmp_path, client)
    assert {"web_search", "web_open", "web_extract"}.issubset(tools.names())
    invalid = tools.execute(
        "web_search",
        {"query": "x", "max_results": 11},
    )
    assert invalid["error"] == "invalid_arguments"
    good = tools.execute("web_search", {"query": "x"})
    assert good["ok"] and good["result_count"] == 1
    records = [
        json.loads(line)
        for line in (tmp_path / "actions.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    assert records[-1]["tool"] == "web_search"


def test_supervisor_can_use_web_tool_then_answer(tmp_path):
    search_html = b"""
    <div class="result"><a class="result__a"
    href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2F">Fonte</a></div>
    """
    web = WebClient(
        WebConfig(
            search_provider="duckduckgo_html",
            search_url="https://html.duckduckgo.com/html/",
        ),
        resolver=public_resolver,
        transport=lambda url: response(url, "text/html", search_html),
    )
    tool_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "web-one",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": json.dumps({"query": "noticia atual"}),
                },
            }
        ],
    }
    model = FakeModel(
        [
            {"choices": [{"message": tool_call}]},
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Fonte encontrada.",
                        }
                    }
                ]
            },
        ]
    )
    result = Supervisor(
        load_settings({"MODEL_MAX_TOOL_CALLS": "3"}),
        model,
        web_registry(tmp_path, web),
    ).run("Pesquise")
    assert result["ok"] and result["tool_calls"] == 1
    assert result["web"] == {"used": True, "sources": []}


def test_supervisor_attaches_verified_opened_sources(tmp_path):
    html = b"<html><head><title>Fonte verificada</title></head><body><p>Fato.</p></body></html>"
    web = WebClient(
        WebConfig(),
        resolver=public_resolver,
        transport=lambda url: response(url, "text/html", html),
    )
    tool_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "open-one",
                "type": "function",
                "function": {
                    "name": "web_open",
                    "arguments": json.dumps(
                        {"url": "https://example.com/source"}
                    ),
                },
            }
        ],
    }
    model = FakeModel(
        [
            {"choices": [{"message": tool_call}]},
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Fato confirmado.",
                        }
                    }
                ]
            },
        ]
    )
    result = Supervisor(
        load_settings({"MODEL_MAX_TOOL_CALLS": "3"}),
        model,
        web_registry(tmp_path, web),
    ).run("Abra e responda")
    assert result["web"]["sources"] == [
        {
            "title": "Fonte verificada",
            "url": "https://example.com/source",
        }
    ]
    assert "[Fonte verificada](https://example.com/source)" in result["answer"]


def test_tool_limit_forces_final_answer_without_more_tools(tmp_path):
    html = b"<html><head><title>Fonte final</title></head><body><p>Fato.</p></body></html>"
    web = WebClient(
        WebConfig(),
        resolver=public_resolver,
        transport=lambda url: response(url, "text/html", html),
    )
    tool_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "last-tool",
                "type": "function",
                "function": {
                    "name": "web_open",
                    "arguments": json.dumps(
                        {"url": "https://example.com/final"}
                    ),
                },
            }
        ],
    }
    model = FakeModel(
        [
            {"choices": [{"message": tool_call}]},
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Resposta final confirmada.",
                        }
                    }
                ]
            },
        ]
    )
    result = Supervisor(
        load_settings({"MODEL_MAX_TOOL_CALLS": "1"}),
        model,
        web_registry(tmp_path, web),
    ).run("Abra e responda")
    assert result["ok"] and result["tool_calls"] == 1
    assert "[Fonte final](https://example.com/final)" in result["answer"]


def test_web_configuration_is_centralized():
    settings = load_settings(
        {
            "WEB_ENABLED": "false",
            "WEB_TIMEOUT": "7",
            "WEB_ALLOWED_DOMAINS": "openai.com,example.com",
            "WEB_BLOCKED_DOMAINS": "ads.example.com",
        }
    )
    assert not settings.web_enabled
    assert settings.web_search_provider == "bing_rss"
    assert settings.web_safe_search == "off"
    assert settings.web_threat_analysis_enabled
    assert settings.web_threat_learning_enabled
    assert settings.web_timeout == 7
    assert settings.web_allowed_domains == ("openai.com", "example.com")
    assert settings.web_blocked_domains == ("ads.example.com",)


def test_web_safe_search_is_configurable_and_validated():
    assert (
        load_settings({"WEB_SAFE_SEARCH": "STRICT"}).web_safe_search
        == "strict"
    )
    with pytest.raises(ValueError, match="WEB_SAFE_SEARCH"):
        load_settings({"WEB_SAFE_SEARCH": "disabled"})
    with pytest.raises(WebError, match="safe_search"):
        WebClient(WebConfig(safe_search="disabled"))


def test_dotenv_is_loaded_for_process_invocation(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "WEB_SEARCH_PROVIDER=duckduckgo_html\n"
        "WEB_SEARCH_URL=https://html.duckduckgo.com/html/\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TERN_ENV_FILE", str(env_file))
    monkeypatch.delenv("WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("WEB_SEARCH_URL", raising=False)
    try:
        settings = load_settings()
        assert settings.env_file == env_file.resolve()
        assert settings.env_file_loaded
        assert settings.web_search_provider == "duckduckgo_html"
    finally:
        os.environ.pop("WEB_SEARCH_PROVIDER", None)
        os.environ.pop("WEB_SEARCH_URL", None)
