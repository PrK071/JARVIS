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
    assert parser.find_by_id("recursos")
    assert parser.find_by_id("como-funciona")
    assert parser.find_by_id("cadastro")

    links = [attrs.get("href") for tag, attrs in parser.tags if tag == "link"]
    scripts = [attrs.get("src") for tag, attrs in parser.tags if tag == "script"]
    assert "./styles.css" in links
    assert "./app.js" in scripts


def test_registration_form_has_accessible_validation_contract() -> None:
    parser = parsed_page()
    form = parser.find_by_id("signup-form")
    assert form and "novalidate" in form[1]

    expected_fields = {
        "name": ("text", "name"),
        "email": ("email", "email"),
        "password": ("password", "new-password"),
        "password-confirmation": ("password", "new-password"),
        "terms": ("checkbox", None),
    }
    for field_id, (field_type, autocomplete) in expected_fields.items():
        element = parser.find_by_id(field_id)
        assert element, f"Campo ausente: {field_id}"
        attrs = element[1]
        assert attrs.get("type") == field_type
        assert "required" in attrs
        assert attrs.get("aria-describedby")
        if autocomplete:
            assert attrs.get("autocomplete") == autocomplete

    status = parser.find_by_id("form-status")
    assert status and status[1].get("aria-live") == "polite"


def test_styles_cover_responsive_and_reduced_motion_layouts() -> None:
    styles = (SITE / "styles.css").read_text(encoding="utf-8")
    assert "@media (max-width: 960px)" in styles
    assert "@media (max-width: 640px)" in styles
    assert "@media (prefers-reduced-motion: reduce)" in styles
    assert ":focus-visible" in styles


def test_form_validation_is_local_and_handles_submit() -> None:
    script = (SITE / "app.js").read_text(encoding="utf-8")
    assert 'form.addEventListener("submit"' in script
    assert "event.preventDefault()" in script
    assert "aria-invalid" in script
    assert "fetch(" not in script
    assert "XMLHttpRequest" not in script
