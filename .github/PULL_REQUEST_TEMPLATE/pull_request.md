## Summary

<!-- One paragraph: what problem this solves and how. -->

## Changes

<!-- Per-area breakdown. Use tables for multi-file changes. -->

| Area | What changed |
|------|-------------|

## Verification

<!-- List the commands actually run and their results. CI runs these checks: -->

```
uv run ruff check .
uv run ruff format --check .
PYTHONPATH=. uv run pytest
PYTHONPATH=. uv run coverage run --source=obesity -m pytest
uv run coverage report --show-missing
PYTHONPATH=. uv run mypy main.py obesity/ tests/ data/eda.py
uv run mypy audit/
uv run mypy experiments/
```

## Files changed

| File | Lines (+/−) | Description |
|------|------------|-------------|

## Artifacts and evidence

<!-- Optional: link the issue and any evidence that exists. -->

Issue: #N

<!-- Only when the change uses OpenSpec: name the specs touched and the archive path. -->

Archived at `openspec/changes/archive/<date>-<change>/`

Specs updated: `openspec/specs/<domain>/spec.md`

## Checklist

- [ ] PR targets `main` and originates from a feature branch; do not use `dev`
- [ ] Linked issue is approved (`status:approved`) and referenced as `Closes #N`
- [ ] `uv run ruff check .` — clean
- [ ] `uv run ruff format --check .` — clean
- [ ] `PYTHONPATH=. uv run coverage run --source=obesity -m pytest` — all passing
- [ ] `uv run coverage report --show-missing` — review per-file coverage
- [ ] `PYTHONPATH=. uv run mypy main.py obesity/ tests/ data/eda.py` — clean
- [ ] `uv run mypy audit/` — clean
- [ ] `uv run mypy experiments/` — clean
- [ ] New behaviour covered by tests
- [ ] Documentation updated when behaviour or configuration changes
