# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Acquire pinned PolyGuardPrompts and prepare separate binary safety targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from resources_servers.multilingual_safety.task_data import PROTOCOL_VERSION, build_safety_messages


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_FPATH = BENCHMARK_DIR / "data" / "polyguard_benchmark.jsonl"
SOURCE_ID = "ToxicityPrompts/PolyGuardPrompts"
SOURCE_REVISION = "c5b466a95b64ff121db4398246b6abb7672696ec"  # pragma: allowlist secret
SOURCE_FILE = "data/test-00000-of-00001.parquet"
SOURCE_SHA256 = "7089d502cc8cdf710671976c9a3d83d65820018d057bbe506def2c1dbcac4e77"  # pragma: allowlist secret
SOURCE_SPLIT = "test"
SOURCE_LICENSE = "CC-BY-4.0"
EXPECTED_SOURCE_ROWS = 29325
EXPECTED_PREPARED_ROWS = 58378
EXPECTED_EXCLUSIONS = 272
CONVERSION_SCHEMA_VERSION = 1
LANGUAGES = {
    "Arabic": "ar",
    "Chinese": "zh",
    "Czech": "cs",
    "Dutch": "nl",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Hindi": "hi",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Polish": "pl",
    "Portuguese": "pt",
    "Russian": "ru",
    "Spanish": "es",
    "Swedish": "sv",
    "Thai": "th",
}
LABEL_FIELDS = ("prompt_label", "response_label", "prompt_harm_label", "response_harm_label", "response_refusal_label")
CATEGORY_FIELDS = ("prompt_categories", "response_categories", "subcategory")
AGREEMENT_FIELDS = ("prompt_harm_agreement", "response_harm_agreement", "response_refusal_agreement")
SOURCE_FIELDS = {
    "prompt",
    "response",
    "id",
    "language",
    "adversarial",
    "prompt_harm_agreement",
    "response_harm_agreement",
    "response_refusal_agreement",
    *LABEL_FIELDS,
    *CATEGORY_FIELDS,
}


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def _selection(values: Sequence[str] | None, *, allowed: set[str], name: str) -> list[str]:
    selected = sorted(allowed) if values is None else list(values)
    if (
        isinstance(values, str)
        or not selected
        or any(not isinstance(value, str) for value in selected)
        or len(set(selected)) != len(selected)
        or set(selected) - allowed
    ):
        raise ValueError(f"{name} must contain unique values from {sorted(allowed)}")
    return selected


def load_source() -> list[dict[str, object]]:
    """Read the immutable source file from Hugging Face's local download cache."""
    from huggingface_hub import hf_hub_download
    from pyarrow.parquet import read_table

    path = Path(hf_hub_download(SOURCE_ID, SOURCE_FILE, repo_type="dataset", revision=SOURCE_REVISION))
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise ValueError(f"PolyGuardPrompts source checksum mismatch: expected {SOURCE_SHA256}, got {actual}")
    records = read_table(path).to_pylist()
    if len(records) != EXPECTED_SOURCE_ROWS or {row["language"] for row in records} != set(LANGUAGES):
        raise ValueError("Pinned PolyGuardPrompts release has unexpected row count or language coverage")
    return records


def convert_rows(
    source_rows: Sequence[Mapping[str, object]],
    *,
    languages: Sequence[str] | None = None,
    classification_tasks: Sequence[str] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Preserve canonical gold labels, legacy annotations, and taskwise exclusions."""
    selected_languages = _selection(languages, allowed=set(LANGUAGES.values()), name="languages")
    selected_tasks = _selection(classification_tasks, allowed={"prompt", "response"}, name="classification_tasks")
    if not source_rows:
        raise ValueError("PolyGuardPrompts source is empty")
    for position, source in enumerate(source_rows):
        if not isinstance(source, Mapping) or set(source) != SOURCE_FIELDS:
            raise ValueError(f"Unexpected PolyGuardPrompts source columns at row {position}")
        if source["id"] is not None and type(source["id"]) is not int:
            raise ValueError(f"Source id must be an integer or null at row {position}")
        for field in ("prompt", "response", "language", *LABEL_FIELDS, *CATEGORY_FIELDS):
            if source[field] is not None and not isinstance(source[field], str):
                raise ValueError(f"Source {field} must be a string or null at row {position}")
        if type(source["adversarial"]) is not bool:
            raise ValueError(f"Source adversarial must be a boolean at row {position}")
        for field in AGREEMENT_FIELDS:
            value = source[field]
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError(f"Source {field} must be a finite number or null at row {position}")
        for field in ("prompt_label", "response_label"):
            if source[field] not in ("safe", "unsafe", None, "", "N/A"):
                raise ValueError(f"Unexpected canonical {field} at source row {position}: {source[field]!r}")
    identities = Counter((row["id"], row["language"]) for row in source_rows)
    rows, exclusions = [], []
    coverage: dict[str, dict[str, object]] = {}
    for position, source in enumerate(source_rows):
        source_language, source_id = source["language"], source["id"]
        if not isinstance(source_language, str) or source_language not in LANGUAGES:
            raise ValueError(f"Unsupported source language at row {position}: {source_language!r}")
        language = LANGUAGES[source_language]
        counts = coverage.setdefault(language, {"source_rows": 0, "selected_source_rows": 0, "tasks": {}})
        counts["source_rows"] += 1
        if language not in selected_languages:
            continue
        counts["selected_source_rows"] += 1
        row_sha = _digest(dict(source))
        source_key = f"{SOURCE_ID}:{SOURCE_SPLIT}:{source_id}:{language}"
        ambiguous_id = source_id is None or identities[(source_id, source_language)] > 1
        if ambiguous_id:
            source_key += f":{position}:{row_sha[:16]}"
        source_labels = {field: source[field] for field in LABEL_FIELDS}
        source_categories = {field: source[field] for field in CATEGORY_FIELDS}
        for task in selected_tasks:
            field = f"{task}_label"
            label = source[field]
            legacy = source[f"{task}_harm_label"]
            legacy_binary = {"harmful": "unsafe", "unharmful": "safe"}.get(legacy)
            task_counts = counts["tasks"].setdefault(
                task, {"eligible": 0, "excluded": 0, "safe": 0, "unsafe": 0, "legacy_disagreements": 0}
            )
            reason = None
            if label in (None, "", "N/A"):
                reason = "missing_or_nonbinary_gold"
            prompt, response = source["prompt"], source["response"]
            if not isinstance(prompt, str) or not prompt.strip():
                reason = "missing_prompt"
            elif task == "response" and (not isinstance(response, str) or not response.strip()):
                reason = "missing_response"
            if reason:
                task_counts["excluded"] += 1
                exclusions.append(
                    {
                        "source_row_position": position,
                        "source_id": source_id,
                        "language": language,
                        "classification_task": task,
                        "gold_field": field,
                        "source_label": label,
                        "reason": reason,
                    }
                )
                continue
            disagreement = legacy_binary is not None and legacy_binary != label
            task_counts["eligible"] += 1
            task_counts[label] += 1
            task_counts["legacy_disagreements"] += int(disagreement)
            sample_id = f"{source_key}:{task}"
            metadata = {
                "classification_task": task,
                "expected_label": label,
                "benchmark": "polyguard",
                "language": language,
                "sample_id": sample_id,
                "source_dataset": SOURCE_ID,
                "source_revision": SOURCE_REVISION,
                "source_split": SOURCE_SPLIT,
                "source_license_metadata": SOURCE_LICENSE,
                "input_protocol": PROTOCOL_VERSION,
                "source_row_position": position,
                "source_id": source_id,
                "source_language": source_language,
                "source_row_sha256": row_sha,
                "source_identity_ambiguous": ambiguous_id,
                "translation_group_id": None if source_id is None else f"{SOURCE_ID}:{SOURCE_SPLIT}:{source_id}",
                "gold_field": field,
                "source_labels": source_labels,
                "source_categories": source_categories,
                "source_agreements": {field: source[field] for field in AGREEMENT_FIELDS},
                "legacy_label_disagreement": disagreement,
                "adversarial": source["adversarial"],
            }
            rows.append(
                {
                    "responses_create_params": {
                        "input": build_safety_messages(
                            prompt=prompt, response=response if task == "response" else None, classification_task=task
                        ),
                        "temperature": 0.0,
                        "max_output_tokens": 512,
                    },
                    "verifier_metadata": metadata,
                }
            )
    if not rows:
        raise ValueError("No binary-eligible tasks remain in the selected PolyGuardPrompts rows")
    return rows, {
        "schema_version": CONVERSION_SCHEMA_VERSION,
        "benchmark": "polyguard",
        "source_dataset": SOURCE_ID,
        "source_revision": SOURCE_REVISION,
        "source_split": SOURCE_SPLIT,
        "source_file": SOURCE_FILE,
        "source_file_sha256": SOURCE_SHA256,
        "source_license_metadata": SOURCE_LICENSE,
        "code_license": "Apache-2.0",
        "source_rows": len(source_rows),
        "prepared_rows": len(rows),
        "input_protocol": PROTOCOL_VERSION,
        "selected_languages": selected_languages,
        "selected_classification_tasks": selected_tasks,
        "is_subset": set(selected_languages) != set(LANGUAGES.values())
        or set(selected_tasks) != {"prompt", "response"},
        "gold_fields": {"prompt": "prompt_label", "response": "response_label"},
        "coverage": coverage,
        "exclusions": exclusions,
    }


def is_prepared_data_current(fpath: Path | str) -> bool:
    """Reuse only a complete pinned native benchmark with matching generated bytes.

    Gym supplies only the output path to this hook, so cached language/task
    subsets are conservatively rebuilt, including when the same subset is requested.
    """
    path = Path(fpath)
    try:
        metadata = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or metadata.get("is_subset") is not False:
            return False
        expected = {
            "schema_version": CONVERSION_SCHEMA_VERSION,
            "benchmark": "polyguard",
            "source_dataset": SOURCE_ID,
            "source_revision": SOURCE_REVISION,
            "source_split": SOURCE_SPLIT,
            "source_file": SOURCE_FILE,
            "source_file_sha256": SOURCE_SHA256,
            "source_license_metadata": SOURCE_LICENSE,
            "source_rows": EXPECTED_SOURCE_ROWS,
            "prepared_rows": EXPECTED_PREPARED_ROWS,
            "input_protocol": PROTOCOL_VERSION,
            "selected_languages": sorted(LANGUAGES.values()),
            "selected_classification_tasks": ["prompt", "response"],
            "gold_fields": {"prompt": "prompt_label", "response": "response_label"},
        }
        if any(metadata.get(field) != value for field, value in expected.items()):
            return False
        coverage, exclusions = metadata.get("coverage"), metadata.get("exclusions")
        if not isinstance(coverage, dict) or set(coverage) != set(LANGUAGES.values()):
            return False
        if not isinstance(exclusions, list) or len(exclusions) != EXPECTED_EXCLUSIONS:
            return False
        per_language = EXPECTED_SOURCE_ROWS // len(LANGUAGES)
        response_eligible = (EXPECTED_PREPARED_ROWS - EXPECTED_SOURCE_ROWS) // len(LANGUAGES)
        for counts in coverage.values():
            if (
                not isinstance(counts, dict)
                or counts.get("source_rows") != per_language
                or counts.get("selected_source_rows") != per_language
            ):
                return False
            tasks = counts.get("tasks")
            if not isinstance(tasks, dict) or set(tasks) != {"prompt", "response"}:
                return False
            for task, expected_eligible in (("prompt", per_language), ("response", response_eligible)):
                task_counts = tasks[task]
                if (
                    not isinstance(task_counts, dict)
                    or task_counts.get("eligible") != expected_eligible
                    or task_counts.get("excluded") != per_language - expected_eligible
                    or type(task_counts.get("safe")) is not int
                    or type(task_counts.get("unsafe")) is not int
                    or task_counts["safe"] + task_counts["unsafe"] != expected_eligible
                ):
                    return False
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest() == metadata.get("prepared_sha256")
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare(
    *,
    languages: Sequence[str] | None = None,
    classification_tasks: Sequence[str] | None = None,
    output_fpath: str | Path = OUTPUT_FPATH,
) -> Path:
    """Write evaluation-only JSONL and a checksum/coverage/exclusion sidecar manifest."""
    rows, metadata = convert_rows(load_source(), languages=languages, classification_tasks=classification_tasks)
    output = Path(output_fpath)
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    metadata["prepared_sha256"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    _write_atomic(output, content)
    _write_atomic(output.with_suffix(".manifest.json"), json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Prepared {len(rows)} PolyGuardPrompts classification tasks at {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", choices=sorted(LANGUAGES.values()))
    parser.add_argument("--tasks", nargs="+", choices=["prompt", "response"])
    parser.add_argument("--output", type=Path, default=OUTPUT_FPATH)
    arguments = parser.parse_args()
    prepare(languages=arguments.languages, classification_tasks=arguments.tasks, output_fpath=arguments.output)
