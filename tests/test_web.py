from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from urllib.parse import urlsplit

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
    WebTooLarge,
)


PUBLIC_IP = "93.184.216.34"


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
        policy.validate_url("http://127.0.0.1/admin")
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
    assert settings.web_timeout == 7
    assert settings.web_allowed_domains == ("openai.com", "example.com")
    assert settings.web_blocked_domains == ("ads.example.com",)


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
