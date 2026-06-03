"""Unit tests for tmux shell-startup readiness waiting."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import ccgram.tmux_manager as tmux_mod
from ccgram.tmux_manager import TmuxManager


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _FakePane:
    def __init__(
        self,
        text_for_time: Callable[[float], str],
        clock: _FakeClock,
        *,
        command: str = "bash",
    ) -> None:
        self._text_for_time = text_for_time
        self._clock = clock
        self.pane_current_command = command
        self.captures = 0

    def capture_pane(self) -> list[str]:
        self.captures += 1
        return self._text_for_time(self._clock.now).splitlines()


def _patch_clock(monkeypatch, clock: _FakeClock) -> None:
    monkeypatch.setattr(tmux_mod.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(tmux_mod.time, "sleep", clock.sleep)


class TestInteractiveShellReady:
    def test_returns_after_prompt_is_stable(self, monkeypatch) -> None:
        clock = _FakeClock()
        _patch_clock(monkeypatch, clock)
        pane = _FakePane(
            lambda now: "running startup" if now < 0.3 else "user@host 1$",
            clock,
        )

        TmuxManager._wait_for_interactive_shell_ready(cast(Any, pane))

        assert 0.5 <= clock.now < 1.0
        assert pane.captures >= 2

    def test_busy_update_output_does_not_trigger_fallback(self, monkeypatch) -> None:
        clock = _FakeClock()
        _patch_clock(monkeypatch, clock)
        pane = _FakePane(
            lambda now: (
                "Checking for updates to latest version..."
                if now < 2.5
                else "user@host 1$"
            ),
            clock,
        )

        TmuxManager._wait_for_interactive_shell_ready(cast(Any, pane))

        assert clock.now >= 2.7

    def test_falls_back_after_quiet_non_prompt_output(self, monkeypatch) -> None:
        clock = _FakeClock()
        _patch_clock(monkeypatch, clock)
        pane = _FakePane(lambda _now: "ready without a conventional prompt", clock)

        TmuxManager._wait_for_interactive_shell_ready(cast(Any, pane))

        assert clock.now >= tmux_mod._SHELL_STARTUP_FALLBACK_STABLE_SECONDS
