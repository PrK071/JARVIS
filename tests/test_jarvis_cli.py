from tern.orchestrator import jarvis


def test_jarvis_without_arguments_opens_ui(monkeypatch):
    received = []
    monkeypatch.setattr(
        jarvis,
        "orchestrator_main",
        lambda arguments: received.append(arguments) or 0,
    )

    assert jarvis.main([]) == 0
    assert received == [["ui"]]


def test_consoleless_gui_entrypoint_opens_ui(monkeypatch):
    received = []
    monkeypatch.setattr(
        jarvis,
        "orchestrator_main",
        lambda arguments: received.append(arguments) or 0,
    )

    assert jarvis.gui_main() == 0
    assert received == [["ui"]]


def test_jarvis_forwards_existing_commands(monkeypatch):
    received = []
    monkeypatch.setattr(
        jarvis,
        "orchestrator_main",
        lambda arguments: received.append(arguments) or 7,
    )

    assert jarvis.main(["status"]) == 7
    assert received == [["status"]]


def test_jarvis_text_option_selects_typed_session(monkeypatch):
    received = []
    monkeypatch.setattr(
        jarvis,
        "orchestrator_main",
        lambda arguments: received.append(arguments) or 0,
    )

    assert jarvis.main(["--text"]) == 0
    assert received == [["text"]]


def test_jarvis_short_text_option_forwards_remaining_arguments(monkeypatch):
    received = []
    monkeypatch.setattr(
        jarvis,
        "orchestrator_main",
        lambda arguments: received.append(arguments) or 0,
    )

    assert jarvis.main(["-t", "--once"]) == 0
    assert received == [["text", "--once"]]


def test_jarvis_codex_alias_opens_shared_tui(monkeypatch):
    received = []
    monkeypatch.setattr(
        jarvis,
        "orchestrator_main",
        lambda arguments: received.append(arguments) or 0,
    )

    assert jarvis.main(["codex"]) == 0
    assert received == [["codex-shared-tui"]]


def test_jarvis_ui_forwards_to_orchestrator_ui(monkeypatch):
    received = []
    monkeypatch.setattr(
        jarvis,
        "orchestrator_main",
        lambda arguments: received.append(arguments) or 0,
    )

    assert jarvis.main(["ui"]) == 0
    assert received == [["ui"]]
