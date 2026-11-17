"""Static checks for invariants that are easy to bypass by accident."""

from pathlib import Path

_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "futureyou"


def test_every_git_subprocess_goes_through_one_helper():
    """Git must be invoked from `_git`, which is where output flags and config are pinned."""
    offenders = []
    for path in sorted(_PACKAGE.glob("*.py")):
        text = path.read_text()
        if "subprocess" in text and path.name != "tools.py":
            offenders.append(path.name)
        if path.name == "tools.py":
            assert text.count("subprocess.run(") == 1
    assert offenders == []
