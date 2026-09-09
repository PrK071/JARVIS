from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))

    def find_by_id(self, element_id: str) -> tuple[str, dict[str, str | None]] | None:
        return next((item for item in self.tags if item[1].get("id") == element_id), None)


def parsed_page() -> PageParser:
    parser = PageParser()
    parser.feed((SITE / "index.html").read_text(encoding="utf-8"))
    return parser


def test_site_assets_and_primary_landmarks_exist() -> None:
    assert (SITE / "index.html").is_file()
    assert (SITE / "styles.css").is_file()
    assert (SITE / "app.js").is_file()

    parser = parsed_page()
    assert parser.find_by_id("conteudo")
    assert parser.find_by_id("sobre")
    assert parser.find_by_id("projetos")
    assert parser.find_by_id("contato")

    links = [attrs.get("href") for tag, attrs in parser.tags if tag == "link"]
    scripts = [attrs.get("src") for tag, attrs in parser.tags if tag == "script"]
    assert "./styles.css" in links
    assert "./app.js" in scripts


def test_personal_identity_and_metadata_are_present() -> None:
    html = (SITE / "index.html").read_text(encoding="utf-8")
    parser = parsed_page()

    assert "Pedro Henrique Teixeira Alves" in html
    assert parser.find_by_id("hero-title")
    assert parser.find_by_id("current-year")
    assert any(tag == "meta" and attrs.get("name") == "description" for tag, attrs in parser.tags)


def test_mobile_navigation_has_accessible_control() -> None:
    parser = parsed_page()
    menu = next((attrs for tag, attrs in parser.tags if tag == "button" and attrs.get("class") == "menu-toggle"), None)
    navigation = parser.find_by_id("site-nav")

    assert menu
    assert menu.get("aria-controls") == "site-nav"
    assert menu.get("aria-expanded") == "false"
    assert navigation and navigation[1].get("aria-label")


def test_styles_cover_responsive_and_reduced_motion_layouts() -> None:
    styles = (SITE / "styles.css").read_text(encoding="utf-8")
    assert "@media (max-width: 900px)" in styles
    assert "@media (max-width: 640px)" in styles
    assert "@media (prefers-reduced-motion: reduce)" in styles
    assert ":focus-visible" in styles


def test_script_handles_menu_reveal_and_current_year_without_network_calls() -> None:
    script = (SITE / "app.js").read_text(encoding="utf-8")
    assert 'menuButton.addEventListener("click"' in script
    assert "IntersectionObserver" in script
    assert "getFullYear" in script
    assert "fetch(" not in script
    assert "XMLHttpRequest" not in script


def test_hosting_build_embeds_the_validated_static_assets() -> None:
    package = (SITE / "package.json").read_text(encoding="utf-8")
    build = (SITE / "build-hosting.mjs").read_text(encoding="utf-8")

    assert '"build": "node build-hosting.mjs"' in package
    assert 'readFile(resolve(root, "index.html"), "utf8")' in build
    assert 'writeFile(resolve(output, "index.js"), worker, "utf8")' in build
