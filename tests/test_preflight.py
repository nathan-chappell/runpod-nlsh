from nlsh.preflight import MissingToolsError, ensure_required_tools, find_missing_tools


def test_find_missing_tools(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda tool: None if tool in {"jq", "qpdf"} else f"/usr/bin/{tool}")
    assert find_missing_tools(["find", "jq", "qpdf"]) == ["jq", "qpdf"]


def test_ensure_required_tools_message(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda tool: None if tool == "jq" else f"/usr/bin/{tool}")
    try:
        ensure_required_tools(["find", "jq"])
    except MissingToolsError as exc:
        text = str(exc)
        assert "Missing required system tools: jq" in text
        assert "Install the matching apt packages: jq" in text
        assert "bootstrap_system_deps.sh" in text
    else:
        raise AssertionError("ensure_required_tools did not raise for a missing binary")
