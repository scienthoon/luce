# Contributing to Luce

Luce is a recipe for turning a sentence about a task into a calibrated decision model. Contributions are welcome:
bug reports, new `convert` presets, new backbones, evaluation results on public benchmarks, documentation.

## Ground rules

- License: Apache-2.0. By contributing you agree your contribution is licensed the same way.
- Keep the core dependency-light. `luce/config.py`, `luce/data.py` and the tests must stay importable without torch.
- Never commit data, checkpoints, logs or API keys. `.gitignore` already excludes `data/`, `checkpoints/`, `logs/`,
  `release/`, `artifacts/`. Teacher keys are read from environment variables only (`LUCE_TEACHER_API_KEY`,
  `OPENAI_API_KEY`, `AI_GATEWAY_API_KEY`); do not add a default teacher.
- Numbers in the README come from `EXPERIMENTS.md`. If you change a number, add the experiment that produced it
  (data, backbone, epochs, seed, hardware, time) there first.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[all]"            # torch, transformers, peft, sentence-transformers, fastapi ...
python -m unittest tests.test_config -q
```

The unit tests are torch-free and run in CI on every pull request. Model-level smoke tests use
`HuggingFaceTB/SmolLM2-135M` on CPU/MPS (see `EXPERIMENTS.md`, "스모크 테스트").

## Pull requests

1. One change per PR. Describe what changed and why; link the experiment section if the change affects numbers.
2. Run the unit tests. If you touched `luce/model.py` or `luce/train.py`, also run a SmolLM2 smoke train/eval and
   paste the `overall:` lines in the PR.
3. New CLI flags need a one-line `help=` string and a mention in `README.md` if user-facing.
4. New `convert` presets must record the dataset's license in the preset docstring and subsample with a seed
   (see `_subsample` in `luce/convert.py`; the banking77 head-truncation bug is why).

## Reporting results

Open an issue titled `results: <task> <backbone>` with the `luce eval` output (calibrated lines, selective-risk table)
and the exact commands. Held-out numbers must use the checkpoint temperature, not a temperature refit on the
held-out set; say which it is.

## Questions

Open a discussion or issue. Korean and English both fine.
