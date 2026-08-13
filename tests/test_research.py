from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

import pytest

from tern.orchestrator.agent import Supervisor
from tern.orchestrator.config import load_settings
from tern.orchestrator.research import (
    classify_research_request,
    classify_source_type,
    generate_query_variants,
    score_result,
    validate_opened_source,
)
from tern.orchestrator.security import ActionLogger, PathPolicy
from tern.orchestrator.tools import ToolRegistry
from tern.orchestrator.web import FetchResponse, WebClient, WebConfig


NOW = datetime(2026, 7, 29, tzinfo=timezone.utc)
REQUEST = (
    "Pesquise uma notícia recente sobre inteligência artificial "
    "e cite a fonte."
)


def public_resolver(_host: str):
    return ["93.184.216.34"]


def rss(items: list[dict[str, str]]) -> bytes:
    values = []
    for item in items:
        values.append(
            "<item>"
            f"<title>{item['title']}</title>"
            f"<link>{item['url'].replace('&', '&amp;')}</link>"
            f"<description>{item.get('snippet', '')}</description>"
            f"<pubDate>{item.get('published_at', '')}</pubDate>"
            "</item>"
        )
    return (
        "<?xml version='1.0'?><rss version='2.0'><channel>"
        + "".join(values)
        + "</channel></rss>"
    ).encode()


def fetch(url: str, data: bytes, content_type: str = "text/xml"):
    return FetchResponse(
        url=url,
        status=200,
        headers={"content-type": content_type},
        data=data,
        duration_ms=5,
    )


def recent_news():
    return {
        "title": "Empresa lança novo modelo de inteligência artificial",
        "url": "https://reuters.com/technology/ai-model",
        "snippet": "Notícia recente sobre inteligência artificial e tecnologia.",
        "published_at": "Tue, 28 Jul 2026 12:00:00 GMT",
    }


def film():
    return {
        "title": "Artificial (2026 film)",
        "url": "https://en.wikipedia.org/wiki/Artificial_(2026_film)",
        "snippet": "Biographical film starring actors and directed by a filmmaker.",
        "published_at": "Mon, 20 Jul 2026 12:00:00 GMT",
    }


def test_news_intent_classification():
    value = classify_research_request(REQUEST)
    assert value.intent == "news"
    assert value.topic == "inteligência artificial"
    assert value.preferred_source_types == ("news", "official")


@pytest.mark.parametrize(
    "text",
    [
        "notícia sobre IA",
        "resultado recente",
        "situação atual do serviço",
        "versão atual do produto",
        "lançamento do modelo",
    ],
)
def test_freshness_detection(text):
    assert classify_research_request(text).requires_freshness


def test_query_expansion_uses_current_date():
    intent = classify_research_request(REQUEST)
    values = generate_query_variants(
        intent, maximum=4, cross_language=True, now=NOW
    )
    assert "julho 2026" in values[0]
    assert len(values) <= 4


def test_portuguese_query_preserves_news_intent():
    values = generate_query_variants(
        classify_research_request(REQUEST),
        maximum=4,
        cross_language=True,
        now=NOW,
    )
    assert "notícias recentes inteligência artificial" in values[0]


def test_complementary_english_query_is_generated():
    values = generate_query_variants(
        classify_research_request(REQUEST),
        maximum=4,
        cross_language=True,
        now=NOW,
    )
    assert any("artificial intelligence latest" in item for item in values)


def test_ambiguous_meanings_are_excluded():
    intent = classify_research_request(REQUEST)
    queries = generate_query_variants(
        intent, maximum=4, cross_language=False, now=NOW
    )
    assert "filmes" in intent.excluded_meanings
    assert all("-filme" in query for query in queries)


def test_relevance_score_contains_transparent_components():
    value = score_result(recent_news(), classify_research_request(REQUEST), now=NOW)
    assert value["final_score"] > 0.8
    assert value["topic_score"] > 0.8
    assert value["intent_score"] == 1
    assert value["reasons"]


def test_wikipedia_is_penalized_for_recent_news():
    value = score_result(film(), classify_research_request(REQUEST), now=NOW)
    assert value["source_type"] == "encyclopedia"
    assert value["ambiguity_penalty"] >= 0.4
    assert value["final_score"] < 0.55


def test_film_is_penalized():
    item = {
        **film(),
        "url": "https://example.com/movies/artificial",
    }
    value = score_result(item, classify_research_request(REQUEST), now=NOW)
    assert value["source_type"] == "entertainment"
    assert value["final_score"] < 0.2


def test_news_article_is_prioritized_over_film():
    intent = classify_research_request(REQUEST)
    assert score_result(recent_news(), intent, now=NOW)["final_score"] > score_result(
        film(), intent, now=NOW
    )["final_score"]


def test_portuguese_ai_topic_matches_english_news():
    intent = classify_research_request(REQUEST)
    item = {
        "title": "AI race changes technology investment",
        "url": "https://reuters.com/technology/ai-race-2026-07-29",
        "snippet": "Companies accelerate artificial intelligence investment.",
        "published_at": "2026-07-29T12:00:00+00:00",
    }
    value = score_result(item, intent, now=NOW)
    assert value["topic_score"] == 1
    assert value["final_score"] >= 0.9


def test_stt_noise_around_ai_news_keeps_canonical_topic():
    value = classify_research_request(
        "Esquise uma notícia recente sobre inteligência artificial e cítia fonte."
    )
    assert value.intent == "news"
    assert value.topic == "inteligência artificial"


def test_stt_corrupted_news_word_still_classifies_news():
    value = classify_research_request(
        "Esquisa o monotice recente sobre inteligência artificial e cita fonte."
    )
    assert value.intent == "news"
    assert value.topic == "inteligência artificial"


def test_stt_corrupted_search_verb_preserves_ambiguity():
    value = classify_research_request("Esquise Artificial.")
    assert value.topic == "Artificial"
    assert value.ambiguous


def test_word_series_in_news_snippet_is_not_entertainment():
    item = {
        "title": "Amazon cuts jobs in artificial intelligence group",
        "url": "https://reuters.com/business/jobs-ai-2026-07-29",
        "snippet": "Latest in a series of smaller reductions across the company.",
        "published_at": "2026-07-29T12:00:00+00:00",
    }
    assert (
        score_result(item, classify_research_request(REQUEST), now=NOW)[
            "source_type"
        ]
        == "news_article"
    )


def test_official_source_is_prioritized():
    item = {
        "title": "OpenAI announces artificial intelligence update",
        "url": "https://openai.com/news/update",
        "snippet": "Official artificial intelligence announcement.",
        "published_at": "2026-07-28T12:00:00+00:00",
    }
    value = score_result(item, classify_research_request(REQUEST), now=NOW)
    assert value["source_type"] == "official_announcement"
    assert value["source_quality_score"] == 1


def test_relevance_threshold_filters_results():
    client = WebClient(
        WebConfig(min_result_relevance=0.99, max_research_corrections=0),
        resolver=public_resolver,
        transport=lambda url: fetch(url, rss([recent_news()])),
    )
    client.begin_research(REQUEST)
    result = client.search(query="inteligência artificial")
    assert result["result_count"] == 0
    assert result["rejected_results"]


def test_opened_news_source_is_validated():
    opened = {
        "title": recent_news()["title"],
        "url": recent_news()["url"],
        "text": "Inteligência artificial " * 30,
        "published_at": "2026-07-28T12:00:00+00:00",
    }
    result = validate_opened_source(
        opened,
        classify_research_request(REQUEST),
        minimum_score=0.65,
        now=NOW,
    )
    assert result["relevant"] and result["supports_query"]


def test_irrelevant_opened_page_is_rejected():
    opened = {
        "title": "Receita de bolo",
        "url": "https://example.com/receita",
        "text": "Farinha e açúcar. " * 30,
        "published_at": "2026-07-28T12:00:00+00:00",
    }
    result = validate_opened_source(
        opened,
        classify_research_request(REQUEST),
        minimum_score=0.65,
        now=NOW,
    )
    assert not result["relevant"]
    assert result["rejection_reason"]


def test_old_news_source_is_rejected():
    opened = {
        "title": "Inteligência artificial em empresas",
        "url": "https://reuters.com/technology/old-ai",
        "text": "Inteligência artificial " * 30,
        "published_at": "2019-01-01T00:00:00+00:00",
    }
    result = validate_opened_source(
        opened,
        classify_research_request(REQUEST),
        minimum_score=0.65,
        now=NOW,
    )
    assert not result["relevant"]
    assert "antiga" in result["rejection_reason"]


def test_corrective_search_runs_after_bad_first_results():
    calls = []

    def transport(url: str):
        calls.append(url)
        return fetch(url, rss([film()] if len(calls) == 1 else [recent_news()]))

    client = WebClient(
        WebConfig(max_research_corrections=2),
        resolver=public_resolver,
        transport=transport,
    )
    client.begin_research(REQUEST)
    result = client.search(query="inteligência artificial")
    assert len(calls) == 2
    assert result["correction_count"] == 1
    assert result["results"][0]["domain"] == "reuters.com"


def test_bing_news_intent_uses_news_rss_endpoint():
    calls = []

    def transport(url: str):
        calls.append(url)
        return fetch(url, rss([recent_news()]))

    client = WebClient(
        WebConfig(search_url="https://www.bing.com/search"),
        resolver=public_resolver,
        transport=transport,
    )
    client.begin_research(REQUEST)
    result = client.search(query="inteligência artificial")
    assert result["result_count"] == 1
    assert urlsplit(calls[0]).path == "/news/search"


def test_opened_news_inherits_rss_date_when_page_has_none():
    calls = []

    def transport(url: str):
        calls.append(url)
        if "bing.com" in url:
            return fetch(url, rss([recent_news()]))
        html = (
            "<html><head><title>Nova tecnologia de inteligência artificial</title>"
            "</head><body><main>"
            + ("Inteligência artificial transforma empresas. " * 20)
            + "</main></body></html>"
        ).encode()
        return FetchResponse(
            url,
            200,
            {"content-type": "text/html; charset=utf-8"},
            html,
        )

    client = WebClient(
        WebConfig(max_research_corrections=0),
        resolver=public_resolver,
        transport=transport,
    )
    client.begin_research(REQUEST)
    found = client.search(query="inteligência artificial")
    opened = client.open(url=found["results"][0]["url"])
    assert opened["published_at"] == "2026-07-28T12:00:00+00:00"
    assert opened["published_at_source"] == "search_result"
    assert opened["accepted_for_citation"]


def test_correction_limit_is_enforced():
    calls = []

    def transport(url: str):
        calls.append(url)
        return fetch(url, rss([film()]))

    client = WebClient(
        WebConfig(max_research_corrections=1),
        resolver=public_resolver,
        transport=transport,
    )
    client.begin_research(REQUEST)
    result = client.search(query="inteligência artificial")
    assert len(calls) == 2
    assert result["result_count"] == 0


def test_query_loop_is_prevented_by_deduplication():
    intent = classify_research_request(REQUEST)
    values = generate_query_variants(
        intent, maximum=10, cross_language=True, now=NOW
    )
    assert len(values) == len({item.casefold() for item in values})


def test_results_are_deduplicated_by_normalized_url_and_title():
    first = recent_news()
    duplicate = {
        **first,
        "url": first["url"] + "?utm_source=test",
    }
    client = WebClient(
        WebConfig(max_research_corrections=0),
        resolver=public_resolver,
        transport=lambda url: fetch(url, rss([first, duplicate])),
    )
    client.begin_research(REQUEST)
    result = client.search(query="inteligência artificial")
    assert result["result_count"] == 1


def test_only_accepted_opened_source_has_citation():
    search_calls = 0

    def transport(url: str):
        nonlocal search_calls
        if urlsplit(url).hostname == "www.bing.com":
            search_calls += 1
            return fetch(url, rss([film(), recent_news()]))
        if "wikipedia.org" in url:
            body = (
                b"<html><head><title>Artificial (2026 film)</title>"
                b"<meta name='date' content='2026-07-20'></head>"
                b"<body><article><p>A film starring actors. "
                + b"Entertainment. " * 30
                + b"</p></article></body></html>"
            )
        else:
            body = (
                b"<html><head><title>Empresa lanca inteligencia artificial</title>"
                b"<meta property='article:published_time' content='2026-07-28'></head>"
                b"<body><article><p>Inteligencia artificial e tecnologia. "
                + b"Inteligencia artificial. " * 30
                + b"</p></article></body></html>"
            )
        return fetch(url, body, "text/html")

    client = WebClient(
        WebConfig(max_research_corrections=0),
        resolver=public_resolver,
        transport=transport,
    )
    client.begin_research(REQUEST)
    result = client.search(query="inteligência artificial")
    assert result["results"][0]["domain"] == "reuters.com"
    bad = client.open(url=film()["url"])
    good = client.open(url=recent_news()["url"])
    assert "citation" not in bad
    assert good["citation"]["url"] == recent_news()["url"]


def test_ambiguous_artificial_requests_clarification_without_network():
    client = WebClient(
        WebConfig(),
        resolver=public_resolver,
        transport=lambda _url: pytest.fail("network should not run"),
    )
    client.begin_research("Pesquise Artificial.")
    result = client.search(query="Artificial")
    assert result["needs_clarification"]
    assert result["result_count"] == 0


def test_general_information_can_use_wikipedia():
    intent = classify_research_request(
        "Pesquise inteligência artificial"
    )
    item = {
        "title": "Inteligência artificial",
        "url": "https://pt.wikipedia.org/wiki/Inteligencia_artificial",
        "snippet": "Inteligência artificial é um campo da computação.",
        "published_at": None,
    }
    assert score_result(item, intent, now=NOW)["final_score"] >= 0.55


def test_rss_date_is_normalized_to_iso():
    values = WebClient._parse_bing_rss(
        rss([recent_news()]), {"provider": "bing_rss"}
    )
    assert values[0]["published_at"] == "2026-07-28T12:00:00+00:00"


def test_rss_description_markup_is_removed():
    item = recent_news()
    item["snippet"] = "&lt;b&gt;Inteligência&lt;/b&gt; artificial"
    values = WebClient._parse_bing_rss(
        rss([item]), {"provider": "bing_rss"}
    )
    assert values[0]["snippet"] == "Inteligência artificial"


def test_artificial_film_regression_selects_real_news():
    calls = []

    def transport(url: str):
        calls.append(url)
        values = [film()] if len(calls) == 1 else [recent_news()]
        return fetch(url, rss(values))

    client = WebClient(
        WebConfig(),
        resolver=public_resolver,
        transport=transport,
    )
    client.begin_research(REQUEST)
    result = client.search(query="inteligência artificial")
    assert all("Artificial_(2026_film)" not in item["url"] for item in result["results"])
    assert result["results"][0]["url"] == recent_news()["url"]
    assert any(
        "Artificial_(2026_film)" in item["url"]
        for item in result["rejected_results"]
    )
