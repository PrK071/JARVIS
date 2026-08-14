from types import SimpleNamespace

from tern.orchestrator.jarvis_ui import JarvisUI, clean_chat_text, result_turn


def test_result_turn_keeps_decision_summary():
    turn = result_turn(
        {
            "ok": True,
            "answer": "Concluído.",
            "decision": {
                "intent": "CODEX_STATUS",
                "reason_code": "active_job_status_query",
            },
        }
    )

    assert turn.speaker == "JARVIS"
    assert turn.text == "Concluído."
    assert turn.detail == "CODEX_STATUS · active_job_status_query"


def test_result_turn_surfaces_failure_without_traceback():
    turn = result_turn({"ok": False, "error": "qwen_indisponivel"})

    assert turn.speaker == "SISTEMA"
    assert turn.text == "qwen_indisponivel"


def test_enter_sends_message():
    ui = object.__new__(JarvisUI)
    calls = []
    ui.send = lambda: calls.append("sent")

    result = ui._send_event(SimpleNamespace(state=0))

    assert calls == ["sent"]
    assert result == "break"


def test_shift_enter_keeps_newline_behavior():
    ui = object.__new__(JarvisUI)
    calls = []
    ui.send = lambda: calls.append("sent")

    result = ui._send_event(SimpleNamespace(state=0x0001))

    assert calls == []
    assert result is None


def test_clean_chat_text_removes_internal_ids_and_markdown_noise():
    text = """**Detalhes da tarefa:**
* **ID da Tarefa:** `aa4721ad-ba5e-4feb-b86b-7d479028e91a`
* **Status:** Concluída
* **Projeto:** `tern` (`D:\\tern`)
* **Thread Disponível:** Sim"""

    assert clean_chat_text(text) == "Detalhes da tarefa:\n• Status: Concluída\n• Projeto: tern (D:\\tern)"
