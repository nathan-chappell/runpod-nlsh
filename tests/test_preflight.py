from nlsh.preflight import MissingToolsError, ensure_required_tools, find_missing_tools


def test_find_missing_tools(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda tool: None if tool in {"mlr", "qpdf"} else f"/usr/bin/{tool}")
    assert find_missing_tools(["ffmpeg", "mlr", "qpdf"]) == ["mlr", "qpdf"]


def test_ensure_required_tools_message(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda tool: None if tool == "mlr" else f"/usr/bin/{tool}")
    try:
        ensure_required_tools(["find", "mlr"])
    except MissingToolsError as exc:
        text = str(exc)
        assert "Missing required system tools: mlr" in text
        assert "Install the matching apt packages: miller" in text
        assert "bootstrap_system_deps.sh" in text
    else:
        raise AssertionError("ensure_required_tools did not raise for a missing binary")
