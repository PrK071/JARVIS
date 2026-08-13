from tern.orchestrator import jarvis


def test_jarvis_without_arguments_starts_voice(monkeypatch):
    received = []
    monkeypatch.setattr(
        jarvis,
        "orchestrator_main",
        lambda arguments: received.append(arguments) or 0,
    )

    assert jarvis.main([]) == 0
    assert received == [["voice"]]


def test_jarvis_forwards_existing_commands(monkeypatch):
    received = []
    monkeypatch.setattr(
        jarvis,
        "orchestrator_main",
        lambda arguments: received.append(arguments) or 7,
    )

    assert jarvis.main(["status"]) == 7
    assert received == [["status"]]
