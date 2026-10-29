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
PYTHONPATH=. uv run mypy .
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
- [ ] `PYTHONPATH=. uv run pytest` — all passing
- [ ] `PYTHONPATH=. uv run mypy .` — clean
- [ ] New behaviour covered by tests
- [ ] Documentation updated when behaviour or configuration changes
