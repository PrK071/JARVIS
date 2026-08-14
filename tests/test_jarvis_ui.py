from tern.orchestrator.jarvis_ui import result_turn


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
