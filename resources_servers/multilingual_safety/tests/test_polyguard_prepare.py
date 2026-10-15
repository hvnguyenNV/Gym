# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.polyguard import prepare as module
from nemo_gym.environment.manifest import load_manifest
from resources_servers.multilingual_safety.app import MultilingualSafetyRunRequest


def source(**changes: object) -> dict[str, object]:
    """Project-authored fixture; contains no downloaded benchmark text."""
    return {
        "prompt": "How can a community library welcome new visitors?",
        "response": "Offer a tour and explain how to borrow books.",
        "id": 123,
        "language": "English",
        "adversarial": False,
        "prompt_label": "safe",
        "response_label": "safe",
        "prompt_harm_label": "unharmful",
        "response_harm_label": "unharmful",
        "response_refusal_label": "compliance",
        "prompt_categories": "",
        "response_categories": "",
        "subcategory": "benign",
        "prompt_harm_agreement": 3.0,
        "response_harm_agreement": 3.0,
        "response_refusal_agreement": 3.0,
        **changes,
    }


def test_native_messages_do_not_leak_labels_and_preserve_canonical_references() -> None:
    record = source(prompt_label="unsafe", prompt_categories="S2")
    rows, manifest = module.convert_rows([record])
    prompt, response = rows
    assert set(prompt) == {"responses_create_params", "verifier_metadata"}
    assert prompt["responses_create_params"]["temperature"] == 0.0
    assert prompt["responses_create_params"]["max_output_tokens"] == 512
    assert prompt["responses_create_params"]["input"] == [{"role": "user", "content": record["prompt"]}]
    assert response["responses_create_params"]["input"] == [
        {"role": "user", "content": record["prompt"]},
        {"role": "assistant", "content": record["response"]},
    ]
    metadata = prompt["verifier_metadata"]
    assert metadata["expected_label"] == "unsafe"
    assert metadata["gold_field"] == "prompt_label"
    assert metadata["source_labels"]["prompt_harm_label"] == "unharmful"
    assert metadata["source_categories"]["prompt_categories"] == "S2"
    assert metadata["legacy_label_disagreement"] is True
    assert metadata["source_agreements"]["prompt_harm_agreement"] == 3.0
    assert len(metadata["source_row_sha256"]) == 64
    assert manifest["coverage"]["en"]["tasks"]["prompt"]["legacy_disagreements"] == 1
    assert response["verifier_metadata"]["expected_label"] == "safe"
    for row in rows:
        validated = MultilingualSafetyRunRequest.model_validate(row)
        assert validated.verifier_metadata.model_dump()["source_labels"] == row["verifier_metadata"]["source_labels"]


def test_response_na_is_excluded_taskwise_without_legacy_fallback() -> None:
    rows, manifest = module.convert_rows([source(response_label="N/A", response_harm_label="harmful")])
    assert len(rows) == 1 and rows[0]["verifier_metadata"]["classification_task"] == "prompt"
    assert manifest["coverage"]["en"]["tasks"]["response"]["excluded"] == 1
    assert manifest["exclusions"] == [
        {
            "source_row_position": 0,
            "source_id": 123,
            "language": "en",
            "classification_task": "response",
            "gold_field": "response_label",
            "source_label": "N/A",
            "reason": "missing_or_nonbinary_gold",
        }
    ]


def test_missing_response_keeps_prompt_and_reports_exact_denominator() -> None:
    rows, manifest = module.convert_rows([source(response=None)])
    assert len(rows) == 1
    assert manifest["exclusions"][0]["reason"] == "missing_response"
    assert manifest["coverage"]["en"]["tasks"]["prompt"]["safe"] == 1


def test_translation_grouping_and_unique_source_ids_are_stable_under_reordering() -> None:
    english, french = source(), source(language="French", prompt="Fixture de bibliothèque.")
    rows, _ = module.convert_rows([english, french])
    reordered, _ = module.convert_rows([french, english])
    assert {r["verifier_metadata"]["sample_id"] for r in rows} == {
        r["verifier_metadata"]["sample_id"] for r in reordered
    }
    assert len({r["verifier_metadata"]["sample_id"] for r in rows}) == 4
    assert len({r["verifier_metadata"]["translation_group_id"] for r in rows}) == 1
    assert {r["verifier_metadata"]["language"] for r in rows} == {"en", "fr"}


def test_repeated_or_missing_source_ids_retain_position_and_content_identity() -> None:
    rows, _ = module.convert_rows([source(), source(prompt="A distinct project fixture."), source(id=None)])
    assert len({r["verifier_metadata"]["sample_id"] for r in rows}) == 6
    assert all(r["verifier_metadata"]["source_identity_ambiguous"] for r in rows)
    assert rows[-1]["verifier_metadata"]["translation_group_id"] is None


def test_language_and_task_selection_keep_source_coverage() -> None:
    rows, manifest = module.convert_rows(
        [source(), source(language="French")], languages=["fr"], classification_tasks=["prompt"]
    )
    assert len(rows) == 1 and rows[0]["verifier_metadata"]["language"] == "fr"
    assert manifest["is_subset"] is True
    assert manifest["coverage"]["en"]["source_rows"] == 1
    assert manifest["coverage"]["en"]["selected_source_rows"] == 0
    assert manifest["selected_classification_tasks"] == ["prompt"]


@pytest.mark.parametrize(
    "changes",
    [
        {"id": True},
        {"language": "Vietnamese"},
        {"prompt_label": "harmful"},
        {"response_label": 0},
        {"prompt": 42},
        {"response": []},
        {"adversarial": 0},
        {"prompt_harm_agreement": True},
        {"response_harm_agreement": float("nan")},
        {"prompt_categories": {}},
        {"prompt_harm_label": 1},
    ],
)
def test_malformed_source_schema_is_rejected(changes) -> None:
    with pytest.raises(ValueError):
        module.convert_rows([source(**changes)])


@pytest.mark.parametrize(
    "arguments",
    [
        {"languages": "en"},
        {"languages": []},
        {"languages": ["en", "en"]},
        {"languages": ["vi"]},
        {"languages": [1]},
        {"classification_tasks": ["refusal"]},
    ],
)
def test_invalid_selection_is_rejected(arguments) -> None:
    with pytest.raises(ValueError):
        module.convert_rows([source()], **arguments)


def test_empty_or_ineligible_source_is_rejected() -> None:
    for records in ([], [source(prompt_label="N/A", response_label="N/A")], [source(prompt=None)]):
        with pytest.raises(ValueError):
            module.convert_rows(records)
    with pytest.raises(ValueError, match="columns"):
        module.convert_rows([{**source(), "new_column": "unexpected"}])


def test_prepare_is_reproducible_and_failed_conversion_preserves_existing_output(tmp_path, monkeypatch) -> None:
    records = [source()]
    monkeypatch.setattr(module, "load_source", lambda: records)
    output = tmp_path / "prepared.jsonl"
    assert module.prepare(output_fpath=output) == output
    original, sidecar = output.read_bytes(), output.with_suffix(".manifest.json").read_bytes()
    manifest = json.loads(sidecar)
    assert manifest["prepared_sha256"] == hashlib.sha256(original).hexdigest()
    assert manifest["source_revision"] == module.SOURCE_REVISION
    assert manifest["prepared_rows"] == 2
    module.prepare(output_fpath=output)
    assert output.read_bytes() == original and output.with_suffix(".manifest.json").read_bytes() == sidecar
    records[:] = [source(prompt_label="unknown")]
    with pytest.raises(ValueError):
        module.prepare(output_fpath=output)
    assert output.read_bytes() == original and output.with_suffix(".manifest.json").read_bytes() == sidecar


def test_acquisition_requires_pinned_checksum_and_full_release(tmp_path, monkeypatch) -> None:
    import huggingface_hub
    import pyarrow.parquet

    path = tmp_path / "cached.parquet"
    path.write_bytes(b"project fixture bytes")
    calls = []
    monkeypatch.setattr(module, "SOURCE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(module, "EXPECTED_SOURCE_ROWS", 1)
    monkeypatch.setattr(module, "LANGUAGES", {"English": "en"})

    def download(*args, **kwargs):
        calls.append((args, kwargs))
        return str(path)

    class Table:
        def to_pylist(self):
            return [source()]

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(pyarrow.parquet, "read_table", lambda _: Table())
    assert module.load_source() == [source()]
    assert calls == [
        ((module.SOURCE_ID, module.SOURCE_FILE), {"repo_type": "dataset", "revision": module.SOURCE_REVISION})
    ]
    path.write_bytes(b"corrupted bytes")
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.load_source()
    monkeypatch.setattr(module, "SOURCE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setattr(module, "EXPECTED_SOURCE_ROWS", 2)
    with pytest.raises(ValueError, match="row count"):
        module.load_source()


def test_manifest_declares_heldout_prepared_classifier_tasks() -> None:
    manifest = load_manifest(Path(module.BENCHMARK_DIR) / "manifest.yaml")
    assert manifest.name == "polyguard" and manifest.prompt_source == "prepared"
    assert manifest.dataset_owner == "polyguard_simple_agent"
    assert manifest.resources_server == "multilingual_safety"
    assert len(manifest.datasets) == 1 and manifest.datasets[0].type == "benchmark"
    assert manifest.datasets[0].prompt_config is None


def test_cache_requires_full_pin_protocol_and_matching_bytes(tmp_path, monkeypatch) -> None:
    records = [source(language=name) for name in module.LANGUAGES]
    monkeypatch.setattr(module, "load_source", lambda: records)
    monkeypatch.setattr(module, "EXPECTED_SOURCE_ROWS", len(records))
    monkeypatch.setattr(module, "EXPECTED_PREPARED_ROWS", 2 * len(records))
    monkeypatch.setattr(module, "EXPECTED_EXCLUSIONS", 0)
    output = tmp_path / "full.jsonl"
    module.prepare(output_fpath=output)
    sidecar = output.with_suffix(".manifest.json")
    original_manifest = sidecar.read_text()
    assert module.is_prepared_data_current(output)
    for field, value in (
        ("source_revision", "another-source"),
        ("input_protocol", "another-protocol"),
        ("schema_version", 0),
        ("is_subset", True),
        ("selected_languages", ["en"]),
        ("selected_classification_tasks", ["prompt"]),
        ("prepared_rows", 1),
        ("gold_fields", {"prompt": "prompt_harm_label", "response": "response_harm_label"}),
        ("coverage", {}),
        ("exclusions", [{"reason": "unexpected"}]),
    ):
        manifest = json.loads(original_manifest)
        manifest[field] = value
        sidecar.write_text(json.dumps(manifest))
        assert not module.is_prepared_data_current(output), field
    sidecar.write_text(original_manifest)
    for modification in ("source_rows", "tasks", "eligible", "safe"):
        manifest = json.loads(original_manifest)
        coverage = manifest["coverage"]["en"]
        if modification == "source_rows":
            coverage["source_rows"] = 0
        elif modification == "tasks":
            coverage["tasks"] = {}
        else:
            coverage["tasks"]["prompt"][modification] = 0
        sidecar.write_text(json.dumps(manifest))
        assert not module.is_prepared_data_current(output), modification
    sidecar.write_text(original_manifest)
    output.write_text(output.read_text() + "\n")
    assert not module.is_prepared_data_current(output)
    module.prepare(languages=["en"], output_fpath=output)
    assert not module.is_prepared_data_current(output)
    for value in ("[]", "{invalid json"):
        sidecar.write_text(value)
        assert not module.is_prepared_data_current(output)
    sidecar.unlink()
    assert not module.is_prepared_data_current(output)
