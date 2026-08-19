from copy import deepcopy
from pathlib import Path

import pytest

from deepks.utils import deep_update, load_yaml, save_yaml
from deepks.scf.fields import select_fields


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPOSITORY_ROOT / "examples"
EXAMPLE_YAML_FILES = tuple(
    path
    for path in sorted(EXAMPLE_ROOT.rglob("*.yaml"))
    if "legacy" not in path.parts
)


@pytest.mark.parametrize(
    "configuration_path",
    EXAMPLE_YAML_FILES,
    ids=lambda path: str(path.relative_to(EXAMPLE_ROOT)),
)
def test_nonlegacy_example_yaml_is_loadable(configuration_path):
    configuration = load_yaml(configuration_path)

    assert isinstance(configuration, dict)
    assert configuration


def test_example_configuration_overlays_merge_as_documented():
    base = load_yaml(EXAMPLE_ROOT / "water_single" / "withdens" / "base.yaml")
    penalty = load_yaml(EXAMPLE_ROOT / "water_single" / "withdens" / "penalty.yaml")
    relax = load_yaml(EXAMPLE_ROOT / "water_single" / "withdens" / "relax.yaml")

    penalized = deep_update(deepcopy(base), penalty)
    relaxed = deep_update(deepcopy(base), relax)

    assert penalized["n_iter"] == 5
    assert penalized["scf_input"]["penalty_terms"][0]["type"] == "coulomb"
    assert relaxed["n_iter"] == 10
    assert "penalty_terms" not in relaxed["scf_input"]


def test_nonlegacy_example_dump_fields_are_registered():
    dump_field_lists = []

    def collect_dump_fields(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "dump_fields":
                    dump_field_lists.append(child)
                collect_dump_fields(child)
        elif isinstance(value, list):
            for child in value:
                collect_dump_fields(child)

    for configuration_path in EXAMPLE_YAML_FILES:
        collect_dump_fields(load_yaml(configuration_path))

    assert dump_field_lists
    for names in dump_field_lists:
        selected = select_fields(names)
        resolved_names = {field.name for group in selected.values() for field in group}
        assert resolved_names == set(names)


def test_yaml_round_trip_supports_a_filename_without_parent(tmp_path, monkeypatch):
    configuration = {
        "method": "deephf",
        "training": {"force_factor": 1.0, "enabled": True},
    }
    monkeypatch.chdir(tmp_path)

    save_yaml(configuration, "configuration.yaml")

    assert load_yaml("configuration.yaml") == configuration
