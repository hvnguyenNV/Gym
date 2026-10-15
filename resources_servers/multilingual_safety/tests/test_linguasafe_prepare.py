# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic tests for LinguaSafe acquisition and label/provenance contracts."""

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from benchmarks.linguasafe import prepare as preparation
from nemo_gym.environment.manifest import load_manifest
from resources_servers.multilingual_safety.app import MultilingualSafetyRunRequest


def source_row(*, level: int = 0, source_id: float | None = None) -> dict[str, object]:
    return {
        "id": source_id,
        "lang": "vi",
        "level": level,
        "prompt": "Hãy giải thích cách giữ an toàn khi đi bộ qua đường.",
        "source": None,
        "specific": True,
        "subtype": "Physical Harm",
        "type": "Harm & Misuse",
    }


@pytest.mark.parametrize(("level", "expected"), [(0, "safe"), (1, "unsafe"), (2, "unsafe"), (3, "unsafe")])
def test_severity_mapping_and_native_input(level: int, expected: str) -> None:
    row = source_row(level=level)
    result = preparation.convert_row(row, row_index=4)
    assert set(result) == {"responses_create_params", "verifier_metadata"}
    assert result["responses_create_params"]["input"] == [{"role": "user", "content": row["prompt"]}]
    metadata = result["verifier_metadata"]
    assert metadata["classification_task"] == "prompt"
    assert metadata["expected_label"] == expected
    assert metadata["severity"] == level
    assert metadata["language"] == "vi"
    assert metadata["specific"] is True
    assert metadata["type"] == row["type"]
    assert metadata["subtype"] == row["subtype"]
    assert metadata["source"] is None
    assert "response_safety" not in metadata
    validated = MultilingualSafetyRunRequest.model_validate(result)
    assert validated.verifier_metadata.model_dump(exclude_unset=True) == metadata


def test_stable_identity_handles_missing_repeated_and_nonfinite_ids() -> None:
    row = source_row(source_id=1.0)
    first = preparation.convert_row(row, row_index=0)["verifier_metadata"]
    assert first == preparation.convert_row(row, row_index=0)["verifier_metadata"]
    assert first["sample_id"] != preparation.convert_row(row, row_index=1)["verifier_metadata"]["sample_id"]
    missing = preparation.convert_row(source_row(), row_index=0)["verifier_metadata"]
    assert missing["source_id"] is None
    assert missing["sample_id"] != first["sample_id"]
    assert missing == preparation.convert_row(source_row(source_id=float("nan")), row_index=0)["verifier_metadata"]
    changed = dict(row, prompt="Project-authored alternative prompt.")
    assert first["sample_id"] != preparation.convert_row(changed, row_index=0)["verifier_metadata"]["sample_id"]
    assert first["source_revision"] == preparation.DATASET_REVISION
    assert first["source_row_index"] == 0
    assert len(first["source_content_sha256"]) == 64


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("prompt", " \n", "prompt must be nonempty"),
        ("prompt", None, "prompt must be nonempty"),
        ("lang", "cz", "unsupported source language"),
        ("level", -1, "level must be an integer"),
        ("level", 4, "level must be an integer"),
        ("level", True, "level must be an integer"),
        ("level", "1", "level must be an integer"),
        ("specific", 1, "specific must be boolean"),
        ("source", "unknown", "source must be an object"),
        ("id", "repeated-id", "id must be numeric"),
        ("id", True, "id must be numeric"),
        ("subtype", "", "subtype must be nonempty"),
        ("type", None, "type must be nonempty"),
    ],
)
def test_invalid_source_contract(field: str, value: object, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        preparation.convert_row(dict(source_row(), **{field: value}), row_index=10)


def test_missing_source_fields_and_negative_index() -> None:
    row = source_row()
    del row["level"]
    with pytest.raises(ValueError, match="missing source fields.*level"):
        preparation.convert_row(row, row_index=0)
    with pytest.raises(ValueError, match="row_index must be nonnegative"):
        preparation.convert_row(source_row(), row_index=-1)


def test_source_object_and_reference_fields_stay_out_of_classifier_input() -> None:
    row = dict(source_row(level=3), source={"dataset": "Synthetic fixture", "content": "original reference"})
    result = preparation.convert_row(row, row_index=1)
    assert result["verifier_metadata"]["source"] == row["source"]
    assert result["responses_create_params"]["input"] == [{"role": "user", "content": row["prompt"]}]


def test_write_dataset_counts_provenance_and_utf8(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preparation, "EXPECTED_LANGUAGE_COUNTS", {"vi": 4})
    monkeypatch.setattr(preparation, "EXPECTED_SEVERITY_COUNTS", {0: 1, 1: 1, 2: 1, 3: 1})
    output = tmp_path / "prepared.jsonl"
    rows = [source_row(level=level) for level in range(4)]
    assert preparation.write_dataset(rows, output_path=output) == output
    tasks = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(tasks) == 4
    assert [task["verifier_metadata"]["severity"] for task in tasks] == [0, 1, 2, 3]
    assert [task["verifier_metadata"]["source_row_index"] for task in tasks] == [0, 1, 2, 3]
    assert "Hãy" in output.read_text(encoding="utf-8")
    provenance = json.loads(output.with_suffix(".manifest.json").read_text())
    assert provenance["row_count"] == 4
    assert provenance["dataset_revision"] == preparation.DATASET_REVISION
    assert provenance["source_split"] == "train"
    assert provenance["dataset_license"] == "CC-BY-NC-SA-4.0"
    assert provenance["protocol"] == preparation.PROTOCOL_VERSION
    assert provenance["label_mapping"] == {"0": "safe", "1": "unsafe", "2": "unsafe", "3": "unsafe"}
    assert provenance["language_counts"] == {"vi": 4}
    assert provenance["severity_counts"] == {"0": 1, "1": 1, "2": 1, "3": 1}
    assert provenance["prepared_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert preparation.is_prepared_data_current(output)
    provenance["protocol"] = "obsolete-classifier-wrapper"
    output.with_suffix(".manifest.json").write_text(json.dumps(provenance))
    assert not preparation.is_prepared_data_current(output)


def test_cache_rejects_missing_invalid_manifest_and_changed_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(preparation, "EXPECTED_LANGUAGE_COUNTS", {"vi": 1})
    monkeypatch.setattr(preparation, "EXPECTED_SEVERITY_COUNTS", {0: 1})
    output = tmp_path / "prepared.jsonl"
    assert not preparation.is_prepared_data_current(output)
    preparation.write_dataset([source_row()], output_path=output)
    assert preparation.is_prepared_data_current(output)
    output.write_text(output.read_text() + "changed data\n")
    assert not preparation.is_prepared_data_current(output)
    output.with_suffix(".manifest.json").write_text("invalid-json")
    assert not preparation.is_prepared_data_current(output)
    output.with_suffix(".manifest.json").write_text("[]")
    assert not preparation.is_prepared_data_current(output)


def test_incomplete_download_preserves_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "prepared.jsonl"
    output.write_text("prior successful preparation\n")
    with pytest.raises(ValueError, match="pinned release counts differ"):
        preparation.write_dataset([source_row()], output_path=output)
    assert output.read_text() == "prior successful preparation\n"
    assert list(tmp_path.iterdir()) == [output]


def test_invalid_row_cleans_temporary_file(tmp_path: Path) -> None:
    output = tmp_path / "prepared.jsonl"
    with pytest.raises(ValueError, match="level must be an integer"):
        preparation.write_dataset([dict(source_row(), level=8)], output_path=output)
    assert list(tmp_path.iterdir()) == []


def test_prepare_requests_immutable_source_and_complete_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import datasets

    rows = [source_row()]
    downloader = Mock(return_value=rows)
    writer = Mock(return_value=tmp_path / "prepared.jsonl")
    monkeypatch.setattr(datasets, "load_dataset", downloader)
    monkeypatch.setattr(preparation, "write_dataset", writer)
    output = preparation.prepare(output_path=str(tmp_path / "prepared.jsonl"))
    downloader.assert_called_once_with(
        preparation.DATASET_ID, revision=preparation.DATASET_REVISION, split=preparation.SOURCE_SPLIT
    )
    writer.assert_called_once_with(rows, output_path=output)


def test_expected_source_counts_cover_all_languages_and_severities() -> None:
    assert set(preparation.EXPECTED_LANGUAGE_COUNTS) == {
        "ar",
        "bn",
        "cs",
        "en",
        "hu",
        "ko",
        "ms",
        "ru",
        "sr",
        "th",
        "vi",
        "zh",
    }
    assert sum(preparation.EXPECTED_LANGUAGE_COUNTS.values()) == 45705
    assert sum(preparation.EXPECTED_SEVERITY_COUNTS.values()) == 45705


def test_manifest_declares_prepared_evaluation_tasks_and_source_license() -> None:
    manifest = load_manifest(preparation.BENCHMARK_DIR / "manifest.yaml")
    assert manifest.name == "linguasafe"
    assert manifest.prompt_source == "prepared"
    assert manifest.canonical_split == "benchmark"
    assert manifest.dataset_owner == "linguasafe_benchmark_simple_agent"
    assert manifest.resources_server == "multilingual_safety"
    assert manifest.licensing == "Apache-2.0 AND CC-BY-NC-SA-4.0"
    assert len(manifest.datasets) == 1
    assert manifest.datasets[0].type == "benchmark"
    assert manifest.datasets[0].prompt_config is None
