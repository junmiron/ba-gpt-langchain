"""Render-path tests for Mermaid: the remote endpoint and the local CLI.

These paths are the ones that decide whether a diagram ships as a vector, a raster or a warning. They are
exercised with a stubbed transport so no network and no headless browser are involved.
"""

from __future__ import annotations

import subprocess

import httpx

from ba_agent import mermaid_utils

VALID = "graph TD\n  A[Start] --> B[Finish]"


class _StubClient:
    """httpx.AsyncClient stand-in that replays one canned response."""

    def __init__(self, *, response=None, error: Exception | None = None):
        self._response = response
        self._error = error

    def __call__(self, **_kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def post(self, url, content, headers):
        if self._error is not None:
            raise self._error
        return self._response


async def test_no_endpoint_means_the_browser_renders_it(monkeypatch):
    svg, error = await mermaid_utils.render_svg(VALID, None)
    assert (svg, error) == (None, None)


async def test_remote_renderer_returning_svg_is_accepted(monkeypatch):
    response = httpx.Response(200, text="<svg xmlns='http://www.w3.org/2000/svg'></svg>")
    monkeypatch.setattr(httpx, "AsyncClient", _StubClient(response=response))
    svg, error = await mermaid_utils.render_svg(VALID, "https://kroki.io")
    assert svg is not None and "<svg" in svg
    assert error is None


async def test_remote_renderer_rejection_is_reported_with_the_status(monkeypatch):
    response = httpx.Response(400, text="Parse error: unsupported syntax")
    monkeypatch.setattr(httpx, "AsyncClient", _StubClient(response=response))
    svg, error = await mermaid_utils.render_svg(VALID, "https://kroki.io")
    assert svg is None
    assert "HTTP 400" in error and "Parse error" in error


async def test_a_non_svg_payload_from_the_renderer_is_not_shipped(monkeypatch):
    response = httpx.Response(200, text="<html><body>not a diagram</body></html>")
    monkeypatch.setattr(httpx, "AsyncClient", _StubClient(response=response))
    svg, error = await mermaid_utils.render_svg(VALID, "https://kroki.io")
    assert svg is None
    assert "non-SVG" in error


async def test_an_unreachable_renderer_degrades_instead_of_raising(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", _StubClient(error=httpx.ConnectError("no route")))
    svg, error = await mermaid_utils.render_svg(VALID, "https://kroki.io")
    assert svg is None
    assert "unreachable" in error


def test_missing_mermaid_cli_reports_it_without_running_anything(monkeypatch, tmp_path):
    monkeypatch.setattr(mermaid_utils, "mermaid_cli", lambda: None)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a))
    svg, png, error = mermaid_utils.render_svg_local(VALID, tmp_path)
    assert svg is None and png is None
    assert error is not None and "not installed" in error
    assert not calls


def _fake_run(*_args, **_kwargs):
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")


def test_local_cli_failure_is_reported_verbatim(monkeypatch, tmp_path):
    monkeypatch.setattr(mermaid_utils, "mermaid_cli", lambda: "mmdc")

    def failing(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="Unsupported type: foo")

    monkeypatch.setattr(subprocess, "run", failing)
    svg, png, error = mermaid_utils.render_svg_local(VALID, tmp_path)
    assert svg is None and png is None
    assert error is not None and "Unsupported type: foo" in error


def test_png_is_only_requested_when_asked_for(monkeypatch, tmp_path):
    monkeypatch.setattr(mermaid_utils, "mermaid_cli", lambda: "mmdc")
    invocations = []

    def recording(args, **kwargs):
        invocations.append(args)
        out = args[4]
        if out.endswith(".svg"):
            (tmp_path / "diagram.svg").write_text("<svg></svg>", encoding="utf-8")
        else:
            (tmp_path / "diagram.png").write_bytes(b"\x89PNG")
        return _fake_run()

    monkeypatch.setattr(subprocess, "run", recording)

    svg, png, error = mermaid_utils.render_svg_local(VALID, tmp_path, want_png=False)
    assert svg == "<svg></svg>" and png is None and error is None
    assert len(invocations) == 1

    svg, png, error = mermaid_utils.render_svg_local(VALID, tmp_path, want_png=True)
    assert svg == "<svg></svg>" and png is not None and png.startswith(b"\x89PNG") and error is None
    # the second call adds one more invocation: the PNG is a separate run, never a `-b` flag.
    assert len(invocations) == 3
    assert any(str(a[4]).endswith(".png") for a in invocations)
