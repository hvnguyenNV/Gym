# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Acquire the pinned LinguaSafe release for derived prompt classification."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile

from resources_servers.multilingual_safety.task_data import PROTOCOL_VERSION, build_safety_messages


BENCHMARK_DIR = Path(__file__).parent
OUTPUT_PATH = BENCHMARK_DIR / "data" / "linguasafe_benchmark.jsonl"
DATASET_ID = "zhiyuan-ning/linguasafe"
DATASET_REVISION = "7cb0bc217efd98293ff715294606522a8625321d"
SOURCE_SPLIT = "train"
DATASET_LICENSE = "CC-BY-NC-SA-4.0"
REFERENCE_EVALUATOR_URL = (
    "https://github.com/zy-ning/LinguaSafe/blob/fd5471728e106b7059674e10bb2ebefc32e6cda7/linguasafe/eval.py"
)
EXPECTED_LANGUAGE_COUNTS = {
    "ar": 4030,
    "bn": 3409,
    "cs": 4057,
    "en": 3330,
    "hu": 3409,
    "ko": 4299,
    "ms": 3407,
    "ru": 4117,
    "sr": 4055,
    "th": 4322,
    "vi": 3884,
    "zh": 3386,
}
EXPECTED_SEVERITY_COUNTS = {0: 5264, 1: 4654, 2: 16559, 3: 19228}
SOURCE_FIELDS = ("id", "lang", "level", "prompt", "source", "specific", "subtype", "type")
LABEL_MAPPING = {"0": "safe", "1": "unsafe", "2": "unsafe", "3": "unsafe"}


def convert_row(row: Mapping[str, object], *, row_index: int) -> dict[str, object]:
    """Map one source prompt to a binary task without exposing its reference label."""
    if row_index < 0:
        raise ValueError("row_index must be nonnegative")
    missing = set(SOURCE_FIELDS).difference(row)
    if missing:
        raise ValueError(f"LinguaSafe row {row_index}: missing source fields {sorted(missing)}")
    prompt = row["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"LinguaSafe row {row_index}: prompt must be nonempty text")
    language = row["lang"]
    if not isinstance(language, str) or language not in EXPECTED_LANGUAGE_COUNTS:
        raise ValueError(f"LinguaSafe row {row_index}: unsupported source language {language!r}")
    level = row["level"]
    if type(level) is not int or level not in EXPECTED_SEVERITY_COUNTS:
        raise ValueError(f"LinguaSafe row {row_index}: level must be an integer from 0 to 3")
    if not isinstance(row["specific"], bool):
        raise ValueError(f"LinguaSafe row {row_index}: specific must be boolean")
    for field in ("type", "subtype"):
        if not isinstance(row[field], str) or not row[field].strip():
            raise ValueError(f"LinguaSafe row {row_index}: {field} must be nonempty text")
    source = row["source"]
    if source is not None and not isinstance(source, dict):
        raise ValueError(f"LinguaSafe row {row_index}: source must be an object or null")
    source_id = row["id"]
    if source_id is not None:
        if isinstance(source_id, bool) or not isinstance(source_id, (int, float)):
            raise ValueError(f"LinguaSafe row {row_index}: id must be numeric or null")
        if not math.isfinite(source_id):
            source_id = None

    canonical_source = {field: row[field] for field in SOURCE_FIELDS}
    canonical_source["id"] = source_id
    content_sha256 = hashlib.sha256(
        json.dumps(canonical_source, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    metadata = {
        "classification_task": "prompt",
        "expected_label": "safe" if level == 0 else "unsafe",
        "benchmark": "linguasafe",
        "language": language,
        "sample_id": f"linguasafe:{DATASET_REVISION}:{row_index}:{content_sha256}",
        "source_dataset": DATASET_ID,
        "source_revision": DATASET_REVISION,
        "source_split": SOURCE_SPLIT,
        "source_license": DATASET_LICENSE,
        "source_id": source_id,
        "source_row_index": row_index,
        "source_content_sha256": content_sha256,
        "severity": level,
        "specific": row["specific"],
        "type": row["type"],
        "subtype": row["subtype"],
        "source": source,
    }
    return {
        "responses_create_params": {
            "input": build_safety_messages(prompt=prompt, classification_task="prompt"),
            "temperature": 0.0,
            "max_output_tokens": 512,
        },
        "verifier_metadata": metadata,
    }


def write_dataset(rows: Iterable[Mapping[str, object]], *, output_path: Path) -> Path:
    """Validate the complete release, then atomically publish its tasks and provenance."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    language_counts: Counter[str] = Counter()
    severity_counts: Counter[int] = Counter()
    count = 0
    digest = hashlib.sha256()
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=output_path.parent, delete=False) as stream:
            temporary_path = Path(stream.name)
            for row_index, row in enumerate(rows):
                task = convert_row(row, row_index=row_index)
                serialized = json.dumps(task, ensure_ascii=False, allow_nan=False) + "\n"
                stream.write(serialized)
                digest.update(serialized.encode("utf-8"))
                language_counts[str(row["lang"])] += 1
                severity_counts[int(row["level"])] += 1
                count += 1
        if dict(language_counts) != EXPECTED_LANGUAGE_COUNTS or dict(severity_counts) != EXPECTED_SEVERITY_COUNTS:
            raise ValueError(
                "LinguaSafe pinned release counts differ from the audited source: "
                f"rows={count}, languages={dict(language_counts)}, severity={dict(severity_counts)}"
            )
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    provenance = {
        "schema_version": 1,
        "benchmark": "linguasafe",
        "task": "derived_binary_prompt_classification",
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "source_split": SOURCE_SPLIT,
        "dataset_license": DATASET_LICENSE,
        "attribution": "Zhiyuan Ning and collaborators, LinguaSafe (2025), https://arxiv.org/abs/2508.12733",
        "modifications": "Each source prompt becomes one task; level 0 is safe and levels 1-3 are unsafe.",
        "reference_evaluator_url": REFERENCE_EVALUATOR_URL,
        "source_url": f"https://huggingface.co/datasets/{DATASET_ID}/tree/{DATASET_REVISION}",
        "label_mapping": LABEL_MAPPING,
        "protocol": PROTOCOL_VERSION,
        "row_count": count,
        "language_counts": dict(sorted(language_counts.items())),
        "severity_counts": dict(sorted(severity_counts.items())),
        "prepared_sha256": digest.hexdigest(),
        "intended_use": "benchmark_evaluation_only",
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_path


def is_prepared_data_current(fpath: Path | str) -> bool:
    """Allow Gym to reuse cached tasks only when their source, protocol, and bytes match."""
    path = Path(fpath)
    try:
        provenance = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        if not isinstance(provenance, dict):
            return False
        if any(
            provenance.get(field) != expected
            for field, expected in {
                "schema_version": 1,
                "task": "derived_binary_prompt_classification",
                "dataset_id": DATASET_ID,
                "dataset_revision": DATASET_REVISION,
                "source_split": SOURCE_SPLIT,
                "dataset_license": DATASET_LICENSE,
                "label_mapping": LABEL_MAPPING,
                "protocol": PROTOCOL_VERSION,
                "row_count": sum(EXPECTED_LANGUAGE_COUNTS.values()),
                "language_counts": EXPECTED_LANGUAGE_COUNTS,
                "severity_counts": {str(level): count for level, count in EXPECTED_SEVERITY_COUNTS.items()},
            }.items()
        ):
            return False
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest() == provenance.get("prepared_sha256")
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False


def prepare(*, output_path: Path | str = OUTPUT_PATH) -> Path:
    """Download the immutable 45,705-row release and retain all twelve languages."""
    from datasets import load_dataset

    source_rows = load_dataset(DATASET_ID, revision=DATASET_REVISION, split=SOURCE_SPLIT)
    output = write_dataset(source_rows, output_path=Path(output_path))
    print(f"Prepared {sum(EXPECTED_LANGUAGE_COUNTS.values())} LinguaSafe prompt tasks at {output}")
    return output


if __name__ == "__main__":
    prepare()
