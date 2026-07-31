from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit


INTENTS = {
    "news",
    "official_documentation",
    "technical_research",
    "general_information",
    "product_information",
    "troubleshooting",
    "academic",
    "current_status",
}

_FRESHNESS_RE = re.compile(
    r"\b(not[ií]cias?|recente[는s]?|hoje|atualiza(?:c[aã]o|ções)|"
    r"vers[aã]o atual|lan[çc]amento|situa[çc][aã]o atual|latest|"
    r"current|today|news|release)\b",
    re.IGNORECASE,
)
_NEWS_RE = re.compile(
    r"\b(?:\w*not[ií]c\w*|news|manchetes?|últimas?)\b",
    re.I,
)
_OFFICIAL_RE = re.compile(
    r"\b(documenta[çc][aã]o oficial|docs? oficial|site oficial|official docs?)\b",
    re.I,
)
_ACADEMIC_RE = re.compile(
    r"\b(artigo acad[eê]mico|paper|estudo cient[ií]fico|pesquisa acad[eê]mica|"
    r"arxiv|doi)\b",
    re.I,
)
_TECHNICAL_RE = re.compile(
    r"\b(documenta[çc][aã]o t[eé]cnica|benchmark|arquitetura|API|SDK|"
    r"implementa[çc][aã]o)\b",
    re.I,
)
_TROUBLESHOOT_RE = re.compile(
    r"\b(erro|falha|problema|corrigir|resolver|troubleshoot|bug)\b", re.I
)
_PRODUCT_RE = re.compile(
    r"\b(pre[çc]o|comprar|produto|modelo|especifica[çc][aã]o|review|"
    r"disponibilidade)\b",
    re.I,
)
_STATUS_RE = re.compile(
    r"\b(status|situa[çc][aã]o atual|est[aá] funcionando|indispon[ií]vel|"
    r"incidente)\b",
    re.I,
)
_SEARCH_FILLER_RE = re.compile(
    r"\b((?:p?esquis\w*|esquis\w*)|buscar?|procure|encontre|sobre|cite|citar|a fonte|"
    r"fontes?|not[ií]cias?|recente[nes]*|hoje|últimas?|latest|news|"
    r"atualiza(?:c[aã]o|ções))\b",
    re.I,
)
_TOKEN_RE = re.compile(r"[\wÀ-ÿ]{2,}", re.UNICODE)
_STOPWORDS = {
    "a",
    "as",
    "ao",
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
    "uma",
    "um",
    "the",
    "and",
    "for",
    "of",
}
_NEWS_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "g1.globo.com",
    "agenciabrasil.ebc.com.br",
    "cnn.com",
    "cnnbrasil.com.br",
    "folha.uol.com.br",
    "estadao.com.br",
    "valor.globo.com",
    "theguardian.com",
    "nytimes.com",
    "technologyreview.com",
    "techcrunch.com",
    "theverge.com",
}
_ENCYCLOPEDIA_DOMAINS = {"wikipedia.org", "britannica.com"}
_FORUM_DOMAINS = {"reddit.com", "stackoverflow.com", "stackexchange.com"}
_SOCIAL_DOMAINS = {"x.com", "twitter.com", "facebook.com", "instagram.com"}
_ACADEMIC_DOMAINS = {"arxiv.org", "doi.org", "nature.com", "science.org"}
_ENTERTAINMENT_RE = re.compile(
    r"\b(filme|film|movie|cinema|livro|book|romance|novel|jogo|game|"
    r"s[eé]rie|series|elenco|cast|diretor|director|biopic)\b",
    re.I,
)
_ENTERTAINMENT_BODY_RE = re.compile(
    r"\b(filme|film|movie|livro|book|jogo|game|biopic)\b|"
    r"\b(?:directed by|dirigido por|estrelado por|stars? .{0,80} as)\b",
    re.I,
)


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return " ".join(
        "".join(char for char in normalized if not unicodedata.combining(char)).split()
    )


def _tokens(value: str) -> set[str]:
    return {
        normalize_text(token)
        for token in _TOKEN_RE.findall(value)
        if normalize_text(token) not in _STOPWORDS
    }


@dataclass(frozen=True)
class ResearchIntent:
    topic: str
    intent: str
    requires_freshness: bool
    language: str
    region: str
    preferred_source_types: tuple[str, ...]
    excluded_meanings: tuple[str, ...]
    ambiguous: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "preferred_source_types": list(self.preferred_source_types),
            "excluded_meanings": list(self.excluded_meanings),
        }


def classify_research_request(
    text: str,
    *,
    language: str = "pt-BR",
    region: str = "BR",
) -> ResearchIntent:
    value = text.strip()
    requires_freshness = bool(_FRESHNESS_RE.search(value))
    if _NEWS_RE.search(value):
        intent = "news"
        preferred = ("news", "official")
    elif _OFFICIAL_RE.search(value):
        intent = "official_documentation"
        preferred = ("official", "documentation")
    elif _ACADEMIC_RE.search(value):
        intent = "academic"
        preferred = ("academic", "official")
    elif _TECHNICAL_RE.search(value):
        intent = "technical_research"
        preferred = ("documentation", "academic", "official")
    elif _TROUBLESHOOT_RE.search(value):
        intent = "troubleshooting"
        preferred = ("official", "documentation", "forum")
    elif _PRODUCT_RE.search(value):
        intent = "product_information"
        preferred = ("official", "commercial", "news")
    elif _STATUS_RE.search(value):
        intent = "current_status"
        preferred = ("official", "news")
        requires_freshness = True
    else:
        intent = "general_information"
        preferred = ("official", "encyclopedia", "documentation")
    topic = _SEARCH_FILLER_RE.sub(" ", value)
    topic = re.sub(r"[?!.:,;\"'()\[\]]+", " ", topic)
    topic = " ".join(topic.split()).strip()
    topic = re.sub(r"^(?:uma?|a|o)\s+", "", topic, flags=re.I)
    topic = re.sub(r"\s+(?:e|com|por)$", "", topic, flags=re.I)
    if not topic:
        topic = value
    normalized_topic = normalize_text(topic)
    ai_topic = (
        "inteligencia artificial" in normalized_topic
        or "artificial intelligence" in normalized_topic
    )
    if intent == "news" and ai_topic:
        topic = (
            "artificial intelligence"
            if "artificial intelligence" in normalized_topic
            else "inteligência artificial"
        )
        normalized_topic = normalize_text(topic)
    excluded = (
        ("filmes", "livros", "jogos", "obras de ficção")
        if intent == "news" and ai_topic
        else ()
    )
    topic_tokens = _tokens(topic)
    ambiguous = (
        len(topic_tokens) <= 1
        and normalized_topic in {"artificial", "ia", "ai"}
        and not ai_topic
    )
    if intent not in INTENTS:
        raise ValueError(f"intencao de pesquisa invalida: {intent}")
    return ResearchIntent(
        topic=topic,
        intent=intent,
        requires_freshness=requires_freshness,
        language=language,
        region=region,
        preferred_source_types=preferred,
        excluded_meanings=excluded,
        ambiguous=ambiguous,
    )


def generate_query_variants(
    classification: ResearchIntent,
    *,
    maximum: int,
    cross_language: bool,
    now: datetime | None = None,
) -> list[str]:
    now = now or datetime.now(timezone.utc)
    topic = classification.topic
    variants: list[str] = []
    if classification.intent == "news":
        month_names = (
            "janeiro",
            "fevereiro",
            "março",
            "abril",
            "maio",
            "junho",
            "julho",
            "agosto",
            "setembro",
            "outubro",
            "novembro",
            "dezembro",
        )
        month = month_names[now.month - 1]
        variants.append(
            f"notícias recentes {topic} {month} {now.year}"
        )
        if cross_language:
            english_topic = normalize_text(topic).replace(
                "inteligencia artificial", "artificial intelligence"
            )
            variants.append(
                f"site:reuters.com {english_topic} latest news "
                f"{now.strftime('%B')} {now.year}"
            )
            variants.append(
                f"{english_topic} latest technology news {now.strftime('%B')} {now.year}"
            )
        variants.extend(
            [
                f"{topic} últimas notícias tecnologia",
                f"{topic} empresas pesquisas regulação notícias",
            ]
        )
    elif classification.intent == "official_documentation":
        variants.extend([f"{topic} documentação oficial", f"{topic} official documentation"])
    elif classification.intent == "academic":
        variants.extend([f"{topic} pesquisa artigo científico", f"{topic} research paper"])
    else:
        variants.append(topic)
        if classification.requires_freshness:
            variants.append(f"{topic} atualização {now.year}")
    exclusions = []
    if classification.excluded_meanings:
        exclusions = ["-filme", "-livro", "-jogo"]
    result = []
    seen = set()
    for query in variants:
        candidate = " ".join([query, *exclusions]).strip()
        if " latest " in f" {query.casefold()} ":
            candidate += " -film -movie -book -game"
        key = normalize_text(candidate)
        if key not in seen:
            seen.add(key)
            result.append(candidate)
        if len(result) >= maximum:
            break
    return result


def parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = value.strip()
    localized = normalize_text(raw)
    localized = re.sub(
        r"^(?:seg|ter|qua|qui|sex|sab|dom)\.?,?\s*",
        "",
        localized,
    )
    month_names = {
        "jan": "Jan",
        "fev": "Feb",
        "mar": "Mar",
        "abr": "Apr",
        "mai": "May",
        "jun": "Jun",
        "jul": "Jul",
        "ago": "Aug",
        "set": "Sep",
        "out": "Oct",
        "nov": "Nov",
        "dez": "Dec",
    }
    for source, target in month_names.items():
        localized = re.sub(
            rf"\b{source}\.?(?=\s)", target, localized, flags=re.I
        )
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(raw)
        except (TypeError, ValueError, OverflowError):
            try:
                parsed = parsedate_to_datetime(localized)
            except (TypeError, ValueError, OverflowError):
                return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def classify_source_type(
    title: str,
    url: str,
    text: str = "",
    *,
    published_at: str | None = None,
) -> str:
    host = (urlsplit(url).hostname or "").casefold()
    path = urlsplit(url).path.casefold()
    title_and_path = f"{title} {path}"
    if any(host == domain or host.endswith("." + domain) for domain in _ENCYCLOPEDIA_DOMAINS):
        return "encyclopedia"
    if _ENTERTAINMENT_RE.search(title_and_path) or _ENTERTAINMENT_BODY_RE.search(
        text[:500]
    ):
        return "entertainment"
    if any(host == domain or host.endswith("." + domain) for domain in _ACADEMIC_DOMAINS):
        return "academic_paper"
    if any(host == domain or host.endswith("." + domain) for domain in _FORUM_DOMAINS):
        return "forum"
    if any(host == domain or host.endswith("." + domain) for domain in _SOCIAL_DOMAINS):
        return "social_media"
    if host.endswith(".gov") or ".gov." in host or host in {
        "openai.com",
        "anthropic.com",
        "microsoft.com",
        "blog.google",
        "deepmind.google",
    }:
        return "official_announcement" if published_at else "documentation"
    if any(host == domain or host.endswith("." + domain) for domain in _NEWS_DOMAINS):
        return "news_article"
    if re.search(r"/(?:news|noticias|article|articles|20\d{2})/", path) and published_at:
        return "news_article"
    if "/docs" in path or "documentation" in normalize_text(title):
        return "documentation"
    if any(value in host for value in ("amazon.", "mercadolivre.", "store.", "shop.")):
        return "commercial"
    return "unknown"


def _topic_score(topic: str, value: str) -> float:
    normalized_topic = normalize_text(topic)
    normalized_value = normalize_text(value)
    if normalized_topic and normalized_topic in normalized_value:
        return 1.0
    if normalized_topic in {
        "inteligencia artificial",
        "artificial intelligence",
    }:
        if (
            "inteligencia artificial" in normalized_value
            or "artificial intelligence" in normalized_value
        ):
            return 1.0
        value_tokens = _tokens(value)
        if "ai" in value_tokens or "ia" in value_tokens:
            return 0.85
    terms = _tokens(topic)
    if not terms:
        return 0.0
    coverage = len(terms & _tokens(value)) / len(terms)
    return min(0.9, coverage * 0.9)


def _freshness_score(published_at: str | None, now: datetime) -> float:
    published = parse_date(published_at)
    if published is None:
        return 0.1
    days = max(0.0, (now - published).total_seconds() / 86400)
    if days <= 2:
        return 1.0
    if days <= 7:
        return 0.95
    if days <= 30:
        return 0.9
    if days <= 90:
        return 0.8
    if days <= 365:
        return 0.5
    return 0.1


def score_result(
    item: dict[str, Any],
    classification: ResearchIntent,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    title = str(item.get("title") or "")
    snippet = str(item.get("snippet") or "")
    url = str(item.get("url") or "")
    published_at = item.get("published_at")
    source_type = classify_source_type(
        title, url, snippet, published_at=published_at
    )
    topic = _topic_score(classification.topic, f"{title} {snippet}")
    if classification.intent == "news":
        intent_scores = {
            "news_article": 1.0,
            "official_announcement": 0.9,
            "academic_paper": 0.55,
            "unknown": 0.35,
            "commercial": 0.2,
            "encyclopedia": 0.05,
            "entertainment": 0.0,
        }
    else:
        intent_scores = {
            "documentation": 1.0,
            "official_announcement": 0.9,
            "academic_paper": 0.85,
            "encyclopedia": 0.75,
            "news_article": 0.7,
            "forum": 0.55,
            "unknown": 0.5,
            "commercial": 0.4,
            "entertainment": 0.25,
        }
    intent_score = intent_scores.get(source_type, 0.3)
    freshness = (
        _freshness_score(published_at, now)
        if classification.requires_freshness
        else 0.7
    )
    quality_scores = {
        "official_announcement": 1.0,
        "documentation": 0.9,
        "academic_paper": 0.9,
        "news_article": 0.85,
        "encyclopedia": 0.65,
        "forum": 0.45,
        "commercial": 0.4,
        "social_media": 0.25,
        "unknown": 0.4,
        "entertainment": 0.2,
    }
    quality = quality_scores.get(source_type, 0.4)
    excluded_hit = any(
        normalize_text(term).split()[0] in normalize_text(f"{title} {snippet}")
        for term in classification.excluded_meanings
    )
    ambiguity_penalty = 0.0
    reasons = []
    if topic >= 0.8:
        reasons.append("contém termos centrais")
    if source_type in {"news_article", "official_announcement"}:
        reasons.append("tipo de fonte atende à intenção")
    if freshness >= 0.8:
        reasons.append("publicação recente")
    if source_type == "encyclopedia" and classification.intent == "news":
        ambiguity_penalty += 0.45
        reasons.append("enciclopédia penalizada para notícia")
    if source_type == "entertainment" or excluded_hit:
        ambiguity_penalty += 0.7
        reasons.append("significado excluído ou entretenimento")
    if classification.intent == "news" and published_at is None:
        ambiguity_penalty += 0.15
        reasons.append("resultado sem data")
    final = (
        0.4 * topic
        + 0.25 * intent_score
        + 0.2 * freshness
        + 0.15 * quality
        - ambiguity_penalty
    )
    return {
        **item,
        "source_type": source_type,
        "topic_score": round(topic, 4),
        "intent_score": round(intent_score, 4),
        "freshness_score": round(freshness, 4),
        "source_quality_score": round(quality, 4),
        "ambiguity_penalty": round(ambiguity_penalty, 4),
        "final_score": round(max(0.0, min(1.0, final)), 4),
        "reasons": reasons,
    }


def validate_opened_source(
    opened: dict[str, Any],
    classification: ResearchIntent,
    *,
    minimum_score: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    title = str(opened.get("title") or "")
    text = str(opened.get("text") or "")
    url = str(opened.get("url") or "")
    published_at = opened.get("published_at")
    source_type = classify_source_type(
        title, url, text, published_at=published_at
    )
    topic_score = _topic_score(classification.topic, f"{title} {text[:12000]}")
    freshness = _freshness_score(published_at, now)
    supports = topic_score >= 0.55 and len(text.strip()) >= 120
    current_enough = (
        not classification.requires_freshness or freshness >= 0.5
    )
    relevant_type = True
    rejection = None
    if classification.intent == "news":
        relevant_type = source_type in {
            "news_article",
            "official_announcement",
        }
        if source_type == "encyclopedia":
            rejection = "enciclopédia não é fonte principal para notícia recente"
        elif source_type == "entertainment":
            rejection = "página de entretenimento não atende ao tópico"
        elif not relevant_type:
            rejection = "tipo de página não atende à intenção de notícia"
        elif not published_at:
            rejection = "notícia sem data identificável"
    if rejection is None and not supports:
        rejection = "conteúdo principal não sustenta o tópico"
    if rejection is None and not current_enough:
        rejection = "fonte antiga demais para a solicitação"
    type_score = 1.0 if relevant_type else 0.25
    score = 0.55 * topic_score + 0.25 * type_score + 0.2 * (
        freshness if classification.requires_freshness else 0.7
    )
    relevant = rejection is None and score >= minimum_score
    if rejection is None and not relevant:
        rejection = "relevância abaixo do limite"
    return {
        "relevant": relevant,
        "relevance_score": round(score, 4),
        "supports_query": supports,
        "is_current_enough": current_enough,
        "source_type": source_type,
        "topic_score": round(topic_score, 4),
        "rejection_reason": rejection,
    }
