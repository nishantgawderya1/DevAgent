# Test fixtures

Repositories DevAgent is pointed at to prove it works end to end. They live here
so the setup is version-controlled and reproducible; each is pushed to GitHub as
its own standalone repo before use.

## `pagination-bug/`

The first end-to-end target. Deliberately small so container installs take
seconds, deliberately safe to spam with PRs and reset between runs.

**The seeded bug spans two files, and that is the point.** `clamp_page_size` in
`pagelib/utils.py` returns `min(requested, maximum + 1)` — off by one. But the
failing tests are in `tests/test_api.py` and they name `paginate`, which lives in
`pagelib/api.py`.

So a character-window RAG system retrieves `paginate`, sees nothing wrong with
it, and has to guess. DevAgent's retriever should surface `paginate` from the
issue text and then pull in `clamp_page_size` as an `expanded` chunk, because
`paginate` calls it. **A successful run here demonstrates that AST dependency
expansion earns its keep** rather than merely proving the plumbing works.

The other five modules (`filters`, `sorting`, `serializers`, `validators`, and
the rest of `utils`) are distractors, so retrieval has to actually choose rather
than trivially returning the whole repo.

### Expected state

| | |
|---|---|
| With the bug | 2 failed, 8 passed |
| Correct fix | 10 passed |
| The fix | `min(requested, maximum + 1)` → `min(requested, maximum)` in `pagelib/utils.py` |

Verify locally before a run: `cd pagination-bug && pytest -q`

### Publishing it

```bash
python scripts/publish_fixture.py pagination-bug --repo <owner>/devagent-fixture
```

Then open [`pagination-bug/ISSUE.md`](pagination-bug/ISSUE.md) as an issue on
that repo and point `/webhook/manual` at its number.

### Resetting between runs

The publish script tags the clean state as `pristine`:

```bash
git checkout pristine && git branch -f main pristine && git push --force origin main
```

Delete the `devagent/issue-N` branches DevAgent leaves behind as you go.
