# Release sanity check

Before pushing the repository, check that the commit only contains source code, configs, examples, documentation, and figures that are intended for public release.

Recommended steps:

```bash
python scripts/check_release.py
git status --short
git diff --cached --name-only
```

Do not push:

- paper PDFs or supplementary PDFs
- slide decks
- private experiment logs
- model checkpoints
- raw benchmark dumps if redistribution is not allowed
- local absolute paths, account names, or project-specific server paths
- identity-revealing metadata files

The two paper figures in `assets/figures/` are intentionally included for the project README.
