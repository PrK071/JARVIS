from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets import Frame, TextArea

from .deepseek import DeepSeekService


HELP_TEXT = """Commands:
/help                 show commands and keyboard shortcuts
/status               local status (no API request)
/new                  create a session for the current project
/sessions [id]        list sessions or select one by short/full id
/project <name>       switch using ProjectRegistry
/history              reload recent persistent messages
/context              inspect the next request estimate
/usage                show authoritative usage accumulated from API responses
/codex [N]            attach a compact, temporary Codex review
/clear-context        remove temporary context only
/model [name]         show model; only the configured model is accepted
/send-codex [confirm] offer/confirm a Qwen-mediated Codex handoff
/exit                 close the TUI

Keys: Enter sends; Alt+Enter inserts a newline; Ctrl+C cancels generation;
Ctrl+D exits when idle. The conversation pane supports mouse/keyboard scrolling."""


LABELS = {
    "human": "Voce",
    "qwen": "Qwen",
    "deepseek": "DeepSeek",
    "codex_context": "Contexto do Codex",
    "system": "Sistema",
}


class DeepSeekTUI:
    """prompt_toolkit presentation layer; it never calls HTTP directly."""

    def __init__(
        self,
        service: DeepSeekService,
        *,
        qwen_handler: Callable[[str], dict[str, Any]] | None = None,
        history_limit: int = 60,
    ):
        self.service = service
        self.qwen_handler = qwen_handler
        self.history_limit = history_limit
        self.messages: list[dict[str, str]] = []
        self._pending_codex_handoff: str | None = None
        self._application: Application | None = None
        self._generation_thread: threading.Thread | None = None
        self.session = self.service.open()
        self.reload_history()

        self.conversation = TextArea(
            text=self._conversation_text(),
            read_only=True,
            scrollbar=True,
            wrap_lines=True,
            focusable=False,
        )
        self.input = TextArea(
            height=3,
            multiline=True,
            prompt="> ",
            wrap_lines=True,
        )
        self.bindings = self._bindings()

    @property
    def application(self) -> Application:
        if self._application is None:
            self._application = self._build_application()
        return self._application

    def _build_application(self) -> Application:
        header = Window(
            FormattedTextControl(self._header),
            height=3,
        )
        status = Window(
            FormattedTextControl(self._footer),
            height=1,
        )
        body = HSplit(
            [
                Frame(header, title="DeepSeek"),
                Frame(self.conversation, title="Conversation"),
                Frame(self.input, title="Message"),
                status,
            ]
        )
        return Application(
            layout=Layout(body, focused_element=self.input),
            key_bindings=self.bindings,
            full_screen=True,
            mouse_support=True,
        )

    def _bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("enter")
        def _send(_event):
            self.submit_current()

        @bindings.add("escape", "enter")
        def _newline(event):
            event.current_buffer.insert_text("\n")

        @bindings.add("c-c")
        def _cancel(_event):
            if self.service.state in {"Generating", "Cancelling"}:
                self.service.cancel()
                self._append("system", "Cancelamento solicitado; resposta parcial sera descartada.")
            else:
                self.input.buffer.reset()

        @bindings.add("c-d")
        def _exit(event):
            if self.service.state not in {"Generating", "Cancelling"}:
                event.app.exit()

        return bindings

    def _header(self) -> FormattedText:
        status = self.service.manager.status(project_path=self.service.project_path)
        project_name = Path(self.service.project_path).name or self.service.project_path
        configured = "Ready" if status.get("configured") and status.get("enabled") else "Offline/config error"
        state = self.service.state if self.service.state != "Ready" else configured
        return FormattedText(
            [
                ("class:title", f" Project: {project_name}    Model: {status.get('model') or '-'}\n"),
                ("", f" Session: {status.get('active_session') or '-'}    State: {state}\n"),
                ("", f" Path: {self.service.project_path}"),
            ]
        )

    def _footer(self) -> FormattedText:
        status = self.service.manager.status(project_path=self.service.project_path)
        report = self.service.context_report()
        estimated = report.get("estimated_total_tokens", 0)
        return FormattedText(
            [
                (
                    "class:status",
                    f" {Path(self.service.project_path).name} | {self.service.state} | "
                    f"{status.get('messages', 0)} messages | ~{estimated} ctx tokens | "
                    "Ctrl+C cancel | /help",
                )
            ]
        )

    def _conversation_text(self) -> str:
        blocks = []
        for message in self.messages:
            label = LABELS.get(message.get("source", "system"), message.get("source", "Sistema"))
            blocks.append(f"{label}\n{message.get('content', '')}")
        return "\n\n".join(blocks)

    def _refresh(self) -> None:
        if hasattr(self, "conversation"):
            self.conversation.text = self._conversation_text()
            self.conversation.buffer.cursor_position = len(self.conversation.text)
        if self._application is not None:
            self._application.invalidate()

    def _append(self, source: str, content: str) -> dict[str, str]:
        value = {"source": source, "content": content}
        self.messages.append(value)
        self._refresh()
        return value

    def reload_history(self) -> None:
        self.messages = [
            {
                "source": str(item.get("source") or "system"),
                "content": str(item.get("content") or ""),
            }
            for item in self.service.manager.history_messages(
                project_path=self.service.project_path,
                limit=self.history_limit,
            )
        ]
        self._refresh()

    def submit_current(self) -> None:
        text = self.input.text.strip()
        if not text:
            return
        self.input.buffer.reset()
        if text.startswith("/"):
            self.handle_command(text)
            return
        if self.service.state in {"Generating", "Cancelling"}:
            self._append("system", "A generation is active; cancel it before sending another message.")
            return
        self._append("human", text)
        placeholder = self._append("deepseek", "")

        def delta(value: str) -> None:
            placeholder["content"] += value
            self._refresh()

        def work() -> None:
            result = self.service.send(text, source="human", on_delta=delta)
            if not result.get("ok"):
                if placeholder in self.messages:
                    self.messages.remove(placeholder)
                self._append(
                    "system",
                    "Generation cancelled; partial response discarded."
                    if result.get("error") == "deepseek_cancelled"
                    else f"DeepSeek unavailable: {result.get('error')}",
                )
            else:
                placeholder["content"] = str(result.get("response") or placeholder["content"])
                self._refresh()

        self._generation_thread = threading.Thread(target=work, daemon=True)
        self._generation_thread.start()

    def handle_command(self, value: str) -> dict[str, Any]:
        parts = value.strip().split(maxsplit=1)
        command = parts[0].casefold()
        argument = parts[1].strip() if len(parts) == 2 else ""
        if command == "/help":
            self._append("system", HELP_TEXT)
            return {"ok": True}
        if command == "/status":
            status = self.service.manager.status(project_path=self.service.project_path)
            text = (
                "DeepSeek Status\n"
                f"Project: {status.get('project')}\nModel: {status.get('model') or '-'}\n"
                f"Session: {status.get('active_session') or '-'}\nMessages: {status.get('messages', 0)}\n"
                f"Context messages: {status.get('context_messages', 0)}\n"
                f"Summary: {'yes' if status.get('summary') else 'no'}\nState: {self.service.state}"
            )
            if not status.get("enabled"):
                text += "\nDeepSeek disabled in configuration."
            elif not self.service.manager.client.api_key:
                text += "\nDeepSeek unavailable: API key not configured."
            self._append("system", text)
            return status
        if command == "/new":
            if self.service.state in {"Generating", "Cancelling"}:
                if argument.casefold() != "confirm":
                    result = {"ok": True, "confirmation_required": True}
                    self._append(
                        "system",
                        "A generation is active. Use /new confirm to cancel it and create a new session.",
                    )
                    return result
                self.service.cancel()
            result = self.service.manager.new_session(self.service.project_path)
            self.session = result
            self.reload_history()
            self._append("system", f"New session: {str(result.get('session_id') or '-')[:8]}")
            return result
        if command == "/sessions":
            if argument:
                result = self.service.manager.switch_session(argument)
                if result.get("ok"):
                    self.service.project_path = str(result["project"])
                    self.session = result
                    self.reload_history()
                else:
                    self._append("system", f"Session selection failed: {result.get('error')}")
                return result
            sessions = self.service.manager.list_sessions()
            lines = ["DeepSeek sessions"]
            for index, item in enumerate(sessions, 1):
                lines.append(
                    f"{index}. {str(item.get('session_id'))[:8]}  {Path(str(item.get('project'))).name}  "
                    f"{item.get('messages', 0)} messages  {item.get('updated_at')}"
                )
            self._append("system", "\n".join(lines) if sessions else "No DeepSeek sessions.")
            return {"ok": True, "sessions": sessions}
        if command == "/project":
            if not argument:
                result = {"ok": False, "error": "project_required"}
            else:
                result = self.service.switch_project(argument)
            if result.get("ok"):
                self.session = result
                self.reload_history()
                self._append(
                    "system",
                    f"Project changed: {Path(self.service.project_path).name}\n{self.service.project_path}\n"
                    f"DeepSeek session: {str(result.get('session_id'))[:8]}",
                )
            else:
                self._append("system", f"Project change failed: {result.get('error')}")
            return result
        if command == "/history":
            self.reload_history()
            return {"ok": True, "messages": len(self.messages)}
        if command == "/context":
            result = self.service.context_report()
            attachments = result.get("attachments") or []
            lines = [
                "Next DeepSeek request (estimated)",
                f"System prompt: ~{result.get('system_prompt', {}).get('estimated_tokens', 0)} tokens",
                f"Rolling summary: ~{result.get('rolling_summary', {}).get('estimated_tokens', 0)} tokens",
                f"Recent conversation: {result.get('recent_conversation', {}).get('messages', 0)} messages, "
                f"~{result.get('recent_conversation', {}).get('estimated_tokens', 0)} tokens",
                f"Temporary context: ~{result.get('temporary_context', {}).get('estimated_tokens', 0)} tokens",
            ]
            lines.extend(
                f"Attached: {item.get('label')} (~{item.get('estimated_tokens')} tokens)"
                for item in attachments
            )
            lines.append(f"Estimated total: ~{result.get('estimated_total_tokens', 0)} input tokens")
            self._append("system", "\n".join(lines))
            return result
        if command == "/usage":
            status = self.service.manager.status(project_path=self.service.project_path)
            usage = status.get("usage") or {}
            self._append(
                "system",
                "Session usage\n"
                f"Requests: {usage.get('requests', 0)}\nInput tokens: {usage.get('input_tokens', 0)}\n"
                f"Output tokens: {usage.get('output_tokens', 0)}\nCached input: {usage.get('cache_hit_tokens', 0)}\n"
                f"Reasoning tokens: {usage.get('reasoning_tokens', 0)}",
            )
            return {"ok": True, "usage": usage}
        if command == "/codex":
            try:
                turns = int(argument or "3")
            except ValueError:
                turns = 0
            if turns <= 0:
                result = {"ok": False, "error": "invalid_turn_limit"}
            else:
                result = self.service.attach_codex(turns)
            if result.get("ok"):
                self._append(
                    "codex_context",
                    f"Attached {result.get('turns', 0)} Codex turns temporarily "
                    f"(~{result.get('estimated_tokens', 0)} tokens).",
                )
            else:
                self._append("system", f"Codex context failed: {result.get('error')}")
            return result
        if command == "/clear-context":
            result = self.service.clear_context()
            self._append("system", f"Temporary context cleared ({result['removed']} attachment(s)).")
            return result
        if command == "/model":
            configured = self.service.manager.client.model
            if not argument or argument == configured:
                result = {"ok": True, "model": configured, "error": None}
                self._append("system", f"Configured model: {configured or '-'}")
            else:
                result = {"ok": False, "error": "deepseek_model_not_allowed"}
                self._append("system", "Model change rejected; configure DEEPSEEK_MODEL and start a new session.")
            return result
        if command == "/send-codex":
            handoff = self.service.qwen_handoff_prompt()
            if not handoff.get("ok"):
                self._append("system", f"Handoff unavailable: {handoff.get('error')}")
                return handoff
            if argument.casefold() != "confirm":
                self._pending_codex_handoff = str(handoff["prompt"])
                self._append(
                    "system",
                    "Send the current DeepSeek recommendation to Qwen for consideration? "
                    "Use /send-codex confirm.",
                )
                return {"ok": True, "confirmation_required": True}
            if self.qwen_handler is None:
                result = {"ok": False, "error": "qwen_handoff_unavailable"}
            else:
                prompt = self._pending_codex_handoff or str(handoff["prompt"])
                result = self.qwen_handler(prompt)
                self._pending_codex_handoff = None
            self._append("qwen", str(result.get("answer") or result.get("message") or result.get("error")))
            return result
        if command == "/exit":
            if self.service.state in {"Generating", "Cancelling"}:
                self.service.cancel()
                self._append("system", "Generation cancellation requested. Use /exit again when cancelled.")
                return {"ok": False, "error": "deepseek_cancelling"}
            if self._application is not None:
                self._application.exit()
            return {"ok": True, "exit": True}
        result = {"ok": False, "error": "unknown_command"}
        self._append("system", "Unknown command. Use /help.")
        return result

    def render_text(self) -> str:
        status = self.service.manager.status(project_path=self.service.project_path)
        return (
            "DeepSeek\n"
            f"Project: {self.service.project_path}\n"
            f"Model: {status.get('model') or '-'}\n"
            f"Session: {status.get('active_session') or '-'}\n"
            f"State: {self.service.state}\n\n"
            + self._conversation_text()
        )

    def run(self) -> None:
        self.application.run()
