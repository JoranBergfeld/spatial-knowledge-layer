Before committing, run `scripts/check`. It is what CI runs, and the pre-push hook runs it too.

The README is the doc. Each endpoint has a `###` section under "The service". After editing
it, run `uv run mdformat README.md` to rewrap instead of wrapping by hand.
