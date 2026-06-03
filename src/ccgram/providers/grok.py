"""Grok Build provider for ccgram.

Grok stores sessions under ``~/.grok/sessions/<urlencoded-cwd>/<session-id>/``.
The user-visible event stream is ``updates.jsonl``. This provider follows the
hookless discovery model used by external sessions: detect a tmux pane running
``grok``, find the newest matching ``updates.jsonl``, then parse new
``session/update`` events into Telegram-facing messages.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any
from urllib.parse import quote

from ccgram.expandable_quote import format_expandable_quote
from ccgram.providers._jsonl import JsonlProvider
from ccgram.providers.base import (
    AgentMessage,
    DiscoveredCommand,
    ProviderCapabilities,
    RESUME_ID_RE,
    SessionStartEvent,
    StatusUpdate,
)
from ccgram.tool_format import format_tool_line

_GROK_BUILTINS: dict[str, str] = {
    "/always-approve": "Toggle auto-approve mode",
    "/clear": "Clear conversation history",
    "/compact": "Compress conversation history",
    "/context": "Show context usage and session stats",
    "/exit": "Exit Grok",
    "/feedback": "Send feedback",
    "/flush": "Flush memory to disk",
    "/hooks": "Open hooks UI",
    "/hooks-add": "Add a custom hook",
    "/hooks-list": "Show loaded hooks",
    "/hooks-remove": "Remove a custom hook",
    "/hooks-trust": "Trust this project for hooks",
    "/hooks-untrust": "Remove hook trust for this project",
    "/mcp": "Show MCP status",
    "/memory": "Browse or manage memory",
    "/model": "Switch model",
    "/new": "Start a new session",
    "/permissions": "Manage permissions",
    "/plan": "Enter plan mode",
    "/plugins": "Manage plugins",
    "/quit": "Exit Grok",
    "/reload-plugins": "Reload plugins",
    "/session-info": "Show session details",
    "/terminal-setup": "Configure terminal support",
    "/tools": "List tools",
}

_TRANSCRIPT_MAX_AGE_SECS = 120.0
_TOOL_RESULT_QUOTE_THRESHOLD = 3
_MAX_TOOL_RESULT_INLINE_CHARS = 1200
_MAX_ARG_SUMMARY = 200


def _update_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    params = entry.get("params")
    if isinstance(params, dict):
        update = params.get("update")
        if isinstance(update, dict):
            return update
    update = entry.get("update")
    if isinstance(update, dict):
        return update
    return entry


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text")
        return text if isinstance(text, str) else ""
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _first_string(data: Any) -> str:
    if isinstance(data, str):
        return data
    if not isinstance(data, dict):
        return ""
    for key in (
        "command",
        "target_file",
        "filePath",
        "path",
        "pattern",
        "query",
        "prompt",
        "description",
        "input",
    ):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    for value in data.values():
        if isinstance(value, str) and value:
            return value
    return ""


def _tool_name(update: dict[str, Any]) -> str:
    title = update.get("title")
    if isinstance(title, str) and title:
        return title
    raw_input = update.get("rawInput")
    if isinstance(raw_input, dict):
        variant = raw_input.get("variant")
        if isinstance(variant, str) and variant:
            return variant
    return "tool"


def _decode_byte_list(output: list[Any]) -> str:
    try:
        return bytes(output).decode("utf-8", errors="replace").strip()
    except TypeError, ValueError:
        return ""


def _format_tool_result(raw_output: Any) -> str:  # noqa: C901
    text = ""
    if isinstance(raw_output, dict):
        for key in (
            "output_for_prompt",
            "tool_output_for_prompt_concise",
            "tool_output_for_prompt",
            "output",
        ):
            value = raw_output.get(key)
            if isinstance(value, str) and value:
                text = value.strip()
                break
        if not text:
            output = raw_output.get("output")
            if isinstance(output, list):
                text = _decode_byte_list(output)
    elif isinstance(raw_output, list):
        text = _decode_byte_list(raw_output)
    elif isinstance(raw_output, str):
        text = raw_output.strip()

    if not text:
        return "Done"

    line_count = text.count("\n") + 1
    if line_count > _TOOL_RESULT_QUOTE_THRESHOLD:
        return f"  {line_count} lines\n{format_expandable_quote(text)}"
    if len(text) > _MAX_TOOL_RESULT_INLINE_CHARS:
        return text[:_MAX_TOOL_RESULT_INLINE_CHARS].rstrip() + "..."
    return text


def _parse_plan(update: dict[str, Any]) -> list[AgentMessage]:
    entries = update.get("entries")
    if not isinstance(entries, list):
        return []
    lines: list[str] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        status = item.get("status", "")
        if isinstance(content, str) and content:
            prefix = f"[{status}] " if isinstance(status, str) and status else ""
            lines.append(prefix + content)
    if not lines:
        return []
    return [
        AgentMessage(
            text="\n".join(lines),
            role="assistant",
            content_type="text",
            phase="plan",
        )
    ]


def _parse_task_completed(update: dict[str, Any]) -> AgentMessage | None:
    snapshot = update.get("task_snapshot")
    if not isinstance(snapshot, dict):
        return None
    command = snapshot.get("command")
    output = snapshot.get("output")
    exit_code = snapshot.get("exit_code")
    parts: list[str] = []
    if isinstance(command, str) and command:
        parts.append(format_tool_line("terminal", command))
    if exit_code is not None:
        parts.append(f"exit: {exit_code}")
    if isinstance(output, str) and output:
        parts.append(_format_tool_result(output))
    if not parts:
        return None
    return AgentMessage(
        text="\n".join(parts),
        role="assistant",
        content_type="tool_result",
    )


def _parse_grok_update(  # noqa: C901, PLR0911
    entry: dict[str, Any],
    pending: dict[str, Any],
) -> tuple[list[AgentMessage], dict[str, Any]]:
    update = _update_from_entry(entry)
    su = update.get("sessionUpdate")
    if not isinstance(su, str):
        return [], pending

    if su == "agent_message_chunk":
        text = _content_text(update.get("content"))
        if not text:
            return [], pending
        return (
            [AgentMessage(text=text, role="assistant", content_type="text")],
            pending,
        )

    if su == "user_message_chunk":
        text = _content_text(update.get("content"))
        if not text:
            return [], pending
        return ([AgentMessage(text=text, role="user", content_type="text")], pending)

    if su == "tool_call":
        tool_call_id = update.get("toolCallId")
        raw_input = update.get("rawInput", {})
        name = _tool_name(update)
        if isinstance(tool_call_id, str) and tool_call_id:
            pending[tool_call_id] = name
        summary = _first_string(raw_input)
        if len(summary) > _MAX_ARG_SUMMARY:
            summary = summary[:_MAX_ARG_SUMMARY] + "..."
        return (
            [
                AgentMessage(
                    text=format_tool_line(name, summary),
                    role="assistant",
                    content_type="tool_use",
                    tool_use_id=tool_call_id if isinstance(tool_call_id, str) else None,
                    tool_name=name,
                )
            ],
            pending,
        )

    if su == "tool_call_update":
        status = update.get("status")
        if status != "completed":
            return [], pending
        tool_call_id = update.get("toolCallId")
        name = (
            pending.pop(tool_call_id, None) if isinstance(tool_call_id, str) else None
        )
        raw_output = update.get("rawOutput")
        if not raw_output:
            return [], pending
        return (
            [
                AgentMessage(
                    text=_format_tool_result(raw_output),
                    role="assistant",
                    content_type="tool_result",
                    tool_use_id=tool_call_id if isinstance(tool_call_id, str) else None,
                    tool_name=name if isinstance(name, str) else None,
                )
            ],
            pending,
        )

    if su == "task_backgrounded":
        command = update.get("command")
        if not isinstance(command, str) or not command:
            return [], pending
        return (
            [
                AgentMessage(
                    text=format_tool_line("terminal", f"background: {command}"),
                    role="assistant",
                    content_type="tool_use",
                    tool_use_id=(
                        update.get("tool_call_id")
                        if isinstance(update.get("tool_call_id"), str)
                        else None
                    ),
                    tool_name="terminal",
                )
            ],
            pending,
        )

    if su == "task_completed":
        message = _parse_task_completed(update)
        return ([message] if message else [], pending)

    if su == "plan":
        return _parse_plan(update), pending

    return [], pending


def _collect_grok_updates(cwd: str) -> list[tuple[float, Path]]:
    encoded = quote(str(Path(cwd).resolve()), safe="")
    root = Path.home() / ".grok" / "sessions" / encoded
    if not root.is_dir():
        return []
    result: list[tuple[float, Path]] = []
    for fpath in root.glob("*/updates.jsonl"):
        try:
            result.append((fpath.stat().st_mtime, fpath))
        except OSError:
            continue
    result.sort(reverse=True)
    return result


def _read_summary(session_dir: Path) -> dict[str, Any] | None:
    fpath = session_dir / "summary.json"
    try:
        data = json.loads(fpath.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _summary_cwd(summary: dict[str, Any]) -> str:
    info = summary.get("info")
    if isinstance(info, dict):
        cwd = info.get("cwd")
        if isinstance(cwd, str):
            return cwd
    cwd = summary.get("cwd")
    return cwd if isinstance(cwd, str) else ""


def _summary_id(summary: dict[str, Any], fallback: str) -> str:
    info = summary.get("info")
    if isinstance(info, dict):
        sid = info.get("id")
        if isinstance(sid, str) and sid:
            return sid
    sid = summary.get("id")
    return sid if isinstance(sid, str) and sid else fallback


class GrokProvider(JsonlProvider):
    """AgentProvider implementation for Grok Build CLI."""

    _CAPS = ProviderCapabilities(
        name="grok",
        launch_command="grok",
        supports_hook=False,
        supports_resume=True,
        supports_continue=True,
        supports_structured_transcript=True,
        builtin_commands=tuple(_GROK_BUILTINS.keys()),
        supports_user_command_discovery=True,
    )

    _BUILTINS = _GROK_BUILTINS

    def make_launch_args(
        self,
        resume_id: str | None = None,
        use_continue: bool = False,
    ) -> str:
        if resume_id:
            if not RESUME_ID_RE.match(resume_id):
                raise ValueError(f"Invalid resume_id: {resume_id!r}")
            return f"--resume {resume_id}"
        if use_continue:
            return "--continue"
        return ""

    def parse_transcript_entries(
        self,
        entries: list[dict[str, Any]],
        pending_tools: dict[str, Any],
        cwd: str | None = None,  # noqa: ARG002
    ) -> tuple[list[AgentMessage], dict[str, Any]]:
        messages: list[AgentMessage] = []
        pending = dict(pending_tools)
        for entry in entries:
            parsed, pending = _parse_grok_update(entry, pending)
            messages.extend(parsed)
        return messages, pending

    def is_user_transcript_entry(self, entry: dict[str, Any]) -> bool:
        return _update_from_entry(entry).get("sessionUpdate") == "user_message_chunk"

    def parse_history_entry(self, entry: dict[str, Any]) -> AgentMessage | None:
        messages, _pending = self.parse_transcript_entries([entry], {})
        return messages[0] if messages else None

    def discover_transcript(
        self,
        cwd: str,
        window_key: str,
        *,
        max_age: float | None = None,
    ) -> SessionStartEvent | None:
        age_limit = _TRANSCRIPT_MAX_AGE_SECS if max_age is None else max_age
        now = time.time()
        resolved_cwd = str(Path(cwd).resolve())
        for mtime, fpath in _collect_grok_updates(resolved_cwd)[:20]:
            if age_limit > 0 and now - mtime > age_limit:
                break
            summary = _read_summary(fpath.parent)
            if summary:
                file_cwd = _summary_cwd(summary)
                if file_cwd and str(Path(file_cwd).resolve()) != resolved_cwd:
                    continue
                session_id = _summary_id(summary, fpath.parent.name)
                event_cwd = file_cwd or resolved_cwd
            else:
                session_id = fpath.parent.name
                event_cwd = resolved_cwd
            return SessionStartEvent(
                session_id=session_id,
                cwd=event_cwd,
                transcript_path=str(fpath),
                window_key=window_key,
            )
        return None

    def parse_terminal_status(
        self,
        pane_text: str,
        *,
        pane_title: str = "",
    ) -> StatusUpdate | None:
        _ = pane_text
        title = pane_title.lower()
        if "working" in title or "streaming" in title:
            return StatusUpdate(raw_text="working", display_label="...working")
        if "ready" in title:
            return StatusUpdate(raw_text="ready", display_label="ready")
        return None

    def discover_commands(self, base_dir: str) -> list[DiscoveredCommand]:
        return super().discover_commands(base_dir)
