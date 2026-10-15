<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# PolyGuardPrompts

This integration evaluates multilingual binary safety classification on
[PolyGuardPrompts](https://huggingface.co/datasets/ToxicityPrompts/PolyGuardPrompts),
the dataset accompanying [Kumar et al. (2025)](https://arxiv.org/abs/2504.04377).
The Gym benchmark name is `polyguard`. This is the multilingual dataset from
`ToxicityPrompts`, distinct from `AI-Secure/PolyGuard`.

The pinned `test` release has 29,325 source interactions: 1,725 in each of 17
languages. Preparation downloads the original Parquet file to the Hugging Face
cache, checks its SHA-256, and writes evaluation tasks locally. Source and
converted benchmark rows are excluded from Git.

## Classification contract

Each interaction produces an independent prompt target and, when its reference
is binary, a response target. The gold fields are exactly `prompt_label` and
`response_label`; `safe` and `unsafe` are preserved without relabeling. The
legacy `prompt_harm_label` and `response_harm_label` fields are retained for
auditing, together with agreement values, refusal labels, categories and
adversarial flags. They never replace missing canonical labels.

| Target | Safe references | Unsafe references | Excluded | Eligible tasks |
| --- | ---: | ---: | ---: | ---: |
| Prompt | 16,677 | 12,648 | 0 | 29,325 |
| Response | 23,919 | 5,134 | 272 (`N/A`) | 29,053 |

The full benchmark contains 58,378 classification tasks. Missing or empty source
responses also exclude only the response task; malformed columns or unknown
canonical labels fail preparation. Every exclusion and language/task/label
denominator is recorded in the generated `.manifest.json` sidecar.

Inputs use the native Nemotron Content Safety exchange: the original user prompt
and, for response classification, the original assistant response. Source text
is preserved exactly. Gold labels and provenance stay in `verifier_metadata`.
The shared [`multilingual_safety`](../../resources_servers/multilingual_safety)
verifier parses native safety verdicts or its documented strict JSON format and
awards `1.0` for a correct binary verdict, `0.0` otherwise. Invalid outputs remain
in the denominator. Per-row generation defaults are temperature `0.0` and
512 output tokens, with one repeat.

This is a canonical-label classifier evaluation. It does not reproduce the
paper's legacy-label aggregate scores, score response refusal, or evaluate
generator jailbreak resistance. Report prompt and response results separately,
including unsafe precision/recall/F1, invalid-output coverage and language support;
use the shared metrics report for pooled and macro language results. Source
categories are preserved without an unreviewed category crosswalk.

## Prepare and run

From the Gym repository with its dependencies installed:

```bash
gym eval prepare --benchmark polyguard

gym eval run --benchmark polyguard \
  --config resources_servers/multilingual_safety/configs/nemotron_content_safety_model.yaml \
  --model MODEL_NAME \
  --model-url http://HOST:PORT/v1 \
  --model-api-key dummy \
  --split benchmark \
  --output results/polyguard/rollouts.jsonl \
  +require_complete=true
```

Use a content-safety model endpoint that applies its own classifier chat
template to user/assistant messages. An ordinary instruction model requires an
explicit classifier rubric; native inputs alone do not ask it to classify.
The example model config preserves all source assistant text and disables classifier
thinking and category generation. Do not add `--model-type vllm_model` alongside it:
loading the base model config again can override its native chat-template settings.
For Nemotron 3.5 revision `35645ed3543b7e7ffaed2e788699e57a5051497c`, a 64K context
window covers the full release: the longest native input contains 43,868 tokens
before the output budget.
No full model baseline is included; the integration remains experimental and
the resources server is `verified: false`.

For a declared language/task subset:

```bash
gym eval prepare --benchmark polyguard \
  '+prepare_script_args={languages:[en,fr],classification_tasks:[prompt]}' \
  +use_cached_prepared_benchmarks=false
```

Supported language codes are `ar`, `zh`, `cs`, `nl`, `en`, `fr`, `de`, `hi`, `it`,
`ja`, `ko`, `pl`, `pt`, `ru`, `es`, `sv`, and `th`. Vietnamese is absent from this
release. Subsets retain complete source coverage in the sidecar and are explicitly
marked `is_subset`; restore the default preparation before a full benchmark run.

## Reproducibility, attribution and dataset rights

- Source: `ToxicityPrompts/PolyGuardPrompts`, split `test`, revision
  `c5b466a95b64ff121db4398246b6abb7672696ec`.
- Source file: `data/test-00000-of-00001.parquet`; SHA-256
  `7089d502cc8cdf710671976c9a3d83d65820018d057bbe506def2c1dbcac4e77`.
- Sample keys include source ID, language and classification target. Translations
  share a `translation_group_id`. Missing or repeated source ID/language pairs
  retain revision, source position and content hash to avoid key collisions.
- Each generated row retains exact source labels, categories, language name, ID,
  position, revision, protocol and row hash. The sidecar includes output checksum,
  selection, coverage, legacy disagreements and the exact exclusions.
- This benchmark derives from WildGuardTest. Group translations together and
  audit overlap with WildGuardTest before splitting data or claiming independent
  evaluation evidence. Keep these held-out records outside prompt optimization
  and model training.

Attribute the dataset to Priyanshu Kumar, Devansh Jain, Akhila Yerukola, Liwei
Jiang, Himanshu Beniwal, Thomas Hartvigsen, and Maarten Sap, *PolyGuard: A
Multilingual Safety Moderation Tool for 17 Languages* (2025), arXiv:2504.04377.

The pinned [dataset card](https://huggingface.co/datasets/ToxicityPrompts/PolyGuardPrompts/blob/c5b466a95b64ff121db4398246b6abb7672696ec/README.md)
declares **CC BY 4.0**. The paper discusses ODC-BY; applicable dataset terms
require clarification before republishing raw or converted records. Code in this
integration is Apache-2.0; that license does not replace source dataset terms or
inherited attribution obligations. This contribution distributes acquisition
code and project-authored test fixtures without third-party benchmark rows.

Preparation uses the standard Hugging Face cache, so a populated cache supports
offline reruns with `HF_HUB_OFFLINE=1`. The source file's byte checksum is checked
again on each preparation. Regenerate prepared data after changing the converter
or native input protocol.
Gym's cache check accepts only the complete pinned release with matching source,
protocol, conversion schema and output checksum. A cached subset triggers
preparation again, even when the same subset is requested; it cannot silently
replace a full benchmark run.

Converter regression tests use project-authored fixtures and run with the shared
resources server's tests:

```bash
pytest resources_servers/multilingual_safety/tests/test_polyguard_prepare.py
```
