# Results conventions (so future-you can write the report fast)

This project is evaluated on clarity and skepticism. These conventions make results reproducible and “reviewable”.

## Directory layout

Store all outputs under `results/`:

- `results/phase<N>/<exp_id>/`
  - `run_config.json` (model, decoding, seeds, device, commit hash)
  - `prompts.jsonl` (exact prompts used)
  - `responses.jsonl` (raw model outputs)
  - `metrics.jsonl` (per-example metrics)
  - `summary.json` (aggregate metrics + small metadata)
  - `notes.md` (any human notes, failures, anomalies)

If you already have a different structure, keep it, but ensure you can trivially extract the above fields.

## Minimal fields for `run_config.json`

- `exp_id`
- `timestamp_utc`
- `git_commit`
- `model_name_or_path`
- `quantization`
- `adapter_path` (or null)
- `dtype`
- `device`
- `seed`
- `decoding`: `{temperature, top_p, max_new_tokens, ...}`
- `scripts`: `{entrypoint, args}`

## Minimal per-example fields

For each example in `metrics.jsonl`:

- `example_id`
- `prompt`
- `raw_output`
- `final_answer` (parsed)
- `think_text` (parsed, if present)
- `metrics`: `{accuracy, refusal, conflict, ...}`
- `parse_warnings` (array)

## What counts as “not cherry-picked”

- Always report:
  - overall accuracy
  - refusal/hedging rate
  - conflict rate (if measuring faithfulness)
- Include:
  - at least 2 strong successes
  - at least 2 failures
  - at least 1 “weird” / ambiguous case

## Plot naming (optional but helpful)

- `fig_<phase>_<exp_id>_<shortdesc>.png`
- If a plot is in the report, it must be traceable to a single experiment ID.


