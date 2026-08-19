import pytest

from deepks.main import main_cli


def test_main_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main_cli(["--help"])

    assert exc_info.value.code == 0
    assert "A program to generate accurate energy functionals." in capsys.readouterr().out
