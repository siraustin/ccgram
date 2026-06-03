import json
import os
import time
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

from ccgram.providers import detect_provider_from_transcript_path
from ccgram.providers.grok import GrokProvider


def _wrap_update(update: dict) -> dict:
    return {"method": "session/update", "params": {"update": update}}


def _write_grok_session(
    home: Path,
    cwd: str,
    session_id: str = "019e8dcc-df0b-7741-b746-53977d3412c9",
    *,
    summary_cwd: str | None = None,
) -> Path:
    encoded = quote(str(Path(cwd).resolve()), safe="")
    session_dir = home / ".grok" / "sessions" / encoded / session_id
    session_dir.mkdir(parents=True)
    (session_dir / "summary.json").write_text(
        json.dumps(
            {
                "info": {
                    "id": session_id,
                    "cwd": summary_cwd if summary_cwd is not None else cwd,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fpath = session_dir / "updates.jsonl"
    fpath.write_text(
        json.dumps(
            _wrap_update(
                {
                    "sessionUpdate": "user_message_chunk",
                    "content": {"type": "text", "text": "Question"},
                }
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return fpath


class TestGrokProviderParsing:
    def test_parses_visible_updates_and_skips_thoughts(self) -> None:
        provider = GrokProvider()
        entries = [
            _wrap_update(
                {
                    "sessionUpdate": "user_message_chunk",
                    "content": {"type": "text", "text": "Question"},
                }
            ),
            _wrap_update(
                {
                    "sessionUpdate": "agent_thought_chunk",
                    "content": {"type": "text", "text": "private reasoning"},
                }
            ),
            _wrap_update(
                {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "Answer"},
                }
            ),
            _wrap_update(
                {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "t1",
                    "title": "Read",
                    "rawInput": {"path": "README.md"},
                }
            ),
            _wrap_update(
                {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "t1",
                    "status": "completed",
                    "rawOutput": {"output_for_prompt": "ok"},
                }
            ),
        ]

        messages, pending = provider.parse_transcript_entries(entries, {})

        assert pending == {}
        assert [(msg.role, msg.content_type) for msg in messages] == [
            ("user", "text"),
            ("assistant", "text"),
            ("assistant", "tool_use"),
            ("assistant", "tool_result"),
        ]
        assert messages[0].text == "Question"
        assert messages[1].text == "Answer"
        assert "README.md" in messages[2].text
        assert messages[3].text == "ok"


class TestGrokProviderDiscovery:
    def test_detects_grok_transcript_path(self) -> None:
        path = "/Users/austin/.grok/sessions/project/session/updates.jsonl"
        assert detect_provider_from_transcript_path(path) == "grok"

    def test_discovers_matching_updates_file(self, tmp_path: Path) -> None:
        cwd = str(tmp_path / "project")
        Path(cwd).mkdir()
        fpath = _write_grok_session(tmp_path, cwd)

        with patch.object(Path, "home", return_value=tmp_path):
            event = GrokProvider().discover_transcript(cwd, "ccgram:@7", max_age=0)

        assert event is not None
        assert event.session_id == "019e8dcc-df0b-7741-b746-53977d3412c9"
        assert event.cwd == cwd
        assert event.transcript_path == str(fpath)
        assert event.window_key == "ccgram:@7"

    def test_skips_summary_cwd_mismatch(self, tmp_path: Path) -> None:
        cwd = str(tmp_path / "project")
        other_cwd = str(tmp_path / "other")
        Path(cwd).mkdir()
        Path(other_cwd).mkdir()
        _write_grok_session(tmp_path, cwd, summary_cwd=other_cwd)

        with patch.object(Path, "home", return_value=tmp_path):
            event = GrokProvider().discover_transcript(cwd, "ccgram:@7", max_age=0)

        assert event is None

    def test_skips_stale_updates_file(self, tmp_path: Path) -> None:
        cwd = str(tmp_path / "project")
        Path(cwd).mkdir()
        fpath = _write_grok_session(tmp_path, cwd)
        old_time = time.time() - 300
        os.utime(fpath, (old_time, old_time))

        with patch.object(Path, "home", return_value=tmp_path):
            event = GrokProvider().discover_transcript(cwd, "ccgram:@7", max_age=100)

        assert event is None

    def test_max_age_zero_ignores_staleness(self, tmp_path: Path) -> None:
        cwd = str(tmp_path / "project")
        Path(cwd).mkdir()
        fpath = _write_grok_session(tmp_path, cwd)
        old_time = time.time() - 300
        os.utime(fpath, (old_time, old_time))

        with patch.object(Path, "home", return_value=tmp_path):
            event = GrokProvider().discover_transcript(cwd, "ccgram:@7", max_age=0)

        assert event is not None
