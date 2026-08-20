import ast
from copy import deepcopy
from pathlib import Path

import pytest

from deepks.utils import deep_update, load_yaml, save_yaml
from deepks.data.fields import select_fields


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPOSITORY_ROOT / "examples"
EXAMPLE_YAML_FILES = tuple(
    path
    for path in sorted(EXAMPLE_ROOT.rglob("*.yaml"))
    if "legacy" not in path.parts
)


def configuration_mappings(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from configuration_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from configuration_mappings(child)


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


def test_force_aware_training_examples_use_only_the_strict_relaxed_contract():
    force_aware_count = 0
    for configuration_path in EXAMPLE_YAML_FILES:
        configuration = load_yaml(configuration_path)
        for mapping in configuration_mappings(configuration):
            data_args = mapping.get("data_args")
            if isinstance(data_args, dict):
                assert (
                    data_args.get("force_name") != "f_corr_explicit_target"
                ), configuration_path
                assert (
                    data_args.get("jacobian_name") != "dq_dR_explicit"
                ), configuration_path
            train_args = mapping.get("train_args")
            if not isinstance(train_args, dict) or train_args.get("force_factor", 0) <= 0:
                continue
            force_aware_count += 1
            assert data_args["force_mode"] == "deephf_relaxed", configuration_path
            assert data_args["force_name"] == "f_corr_target", configuration_path
            assert data_args["jacobian_name"] == "dq_dR_relaxed", configuration_path
    assert force_aware_count == 2


def test_rhf_force_example_has_a_runnable_python_producer_and_train_config():
    producer_path = EXAMPLE_ROOT / "deephf" / "generate_rhf_force_data.py"
    syntax_tree = ast.parse(producer_path.read_text(encoding="utf-8"))
    imported_names = {
        alias.name
        for node in ast.walk(syntax_tree)
        if isinstance(node, ast.ImportFrom) and node.module == "deepks.deephf"
        for alias in node.names
    }
    configuration = load_yaml(EXAMPLE_ROOT / "deephf" / "rhf_force_train.yaml")

    assert "write_rhf_force_dataset" in imported_names
    assert configuration["data_args"] == {
        "batch_size": 1,
        "group_batch": 1,
        "energy_name": "e_corr_target",
        "descriptor_name": "descriptor",
        "force_name": "f_corr_target",
        "jacobian_name": "dq_dR_relaxed",
        "force_mode": "deephf_relaxed",
    }
    assert configuration["train_args"]["force_factor"] > 0
    assert configuration["projector_basis"] == [
        [0, [0.8, 1.0]],
        [1, [0.3, 1.0]],
    ]


def test_yaml_round_trip_supports_a_filename_without_parent(tmp_path, monkeypatch):
    configuration = {
        "method": "deephf",
        "training": {"force_factor": 1.0, "enabled": True},
    }
    monkeypatch.chdir(tmp_path)

    save_yaml(configuration, "configuration.yaml")

    assert load_yaml("configuration.yaml") == configuration
