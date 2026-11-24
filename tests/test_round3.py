"""Round-3 review regressions. Behavior tests failed on fade1cd before the fix."""

import subprocess
from pathlib import Path

import pytest
from conftest import git

from futureyou import cli, tools


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "README.md").write_text("base\n")
    git(repo, "add", "README.md")
    git(repo, "commit", "-q", "-m", "base")
    return repo


def _writer(path: Path, marker: Path) -> None:
    path.write_text(
        "#!/bin/sh\n"
        f"mkdir -p {marker.parent}\n"
        f"echo ran > {marker}\n"
        "printf 'POINTER\\n'\n"
    )
    path.chmod(0o755)


def _drop_git_config_env(env: dict) -> dict:
    cleaned = dict(env)
    for key in list(cleaned):
        text = key.decode() if isinstance(key, bytes) else key
        if text == "GIT_CONFIG_COUNT" or text.startswith(
            ("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")
        ):
            del cleaned[key]
    return cleaned


def _as_git_version(version: str):
    real = tools._git

    def routed(repo, *args, **kwargs):
        if args[:1] == ("version",):
            return subprocess.CompletedProcess(
                ["git", "version"], 0, f"git version {version}\n", ""
            )
        return real(repo, *args, **kwargs)

    return routed


def test_dotted_filter_is_disabled_by_dash_c(tmp_path, monkeypatch):
    """-c must disable a dotted driver even when GIT_CONFIG_COUNT is removed."""
    repo = _repo(tmp_path)
    (repo / "tracked.txt").write_text("v1\n")
    git(repo, "add", "tracked.txt")
    git(repo, "commit", "-q", "-m", "add")
    script = tmp_path / "clean.sh"
    marker = repo / ".git" / "evil" / "ran"
    _writer(script, marker)
    git(repo, "config", "filter.foo.bar.clean", str(script))
    git(repo, "config", "filter.foo.bar.required", "true")
    (repo / ".gitattributes").write_text("* filter=foo.bar\n")
    (repo / "tracked.txt").write_text("v2\n")
    commands: list[list[str]] = []
    real = subprocess.Popen

    def spy(cmd, **kwargs):
        commands.append(list(cmd))
        env = kwargs.get("env")
        if env is not None:
            kwargs["env"] = _drop_git_config_env(env)
        return real(cmd, **kwargs)

    monkeypatch.setattr(tools.subprocess, "Popen", spy)
    result = tools.git_diff(repo)
    assert result["status"] == "ok", result
    assert "+v2\n" in result["diff"]
    assert "POINTER" not in result["diff"]
    assert not marker.exists()
    assert any("filter.foo.bar.clean=" in cmd for cmd in commands), commands


def test_git_230_refuses_a_filter_name_dash_c_cannot_express(
    tmp_path, monkeypatch, capsys
):
    repo = _repo(tmp_path)
    (repo / "tracked.txt").write_text("v1\n")
    git(repo, "add", "tracked.txt")
    git(repo, "commit", "-q", "-m", "add")
    script = tmp_path / "clean.sh"
    marker = repo / ".git" / "evil" / "ran"
    _writer(script, marker)
    git(repo, "config", "filter.foo=bar.clean", str(script))
    git(repo, "config", "filter.foo=bar.required", "true")
    (repo / ".gitattributes").write_text("* filter=foo=bar\n")
    (repo / "tracked.txt").write_text("v2\n")
    monkeypatch.setattr(tools, "_git", _as_git_version("2.30.0"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("FUTUREYOU_MODEL", raising=False)
    with pytest.raises(tools.RepoError, match="2.30"):
        tools.git_diff(repo)
    assert cli.main(["review", str(repo), "--thread", "old-git"]) == 1
    err = capsys.readouterr().err
    assert "model call failed" not in err
    assert "FUTUREYOU_MODEL" not in err
    assert "foo=bar" in err
    assert not marker.exists()


def test_git_230_refuses_a_non_utf8_filter_name(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    (repo / "tracked.txt").write_text("v1\n")
    git(repo, "add", "tracked.txt")
    git(repo, "commit", "-q", "-m", "add")
    script = tmp_path / "clean.sh"
    marker = repo / ".git" / "evil" / "ran"
    _writer(script, marker)
    driver = b"foo" + bytes([0xFF]) + b"bar"
    config = (repo / ".git" / "config").read_bytes()
    (repo / ".git" / "config").write_bytes(
        config
        + b'\n[filter "'
        + driver
        + b'"]\n\tclean = '
        + str(script).encode()
        + b"\n\trequired = true\n"
    )
    (repo / ".gitattributes").write_bytes(b"* filter=" + driver + b"\n")
    (repo / "tracked.txt").write_text("v2\n")
    monkeypatch.setattr(tools, "_git", _as_git_version("2.30.0"))
    with pytest.raises(tools.RepoError, match="2.30"):
        tools.git_diff(repo)
    assert not marker.exists()
