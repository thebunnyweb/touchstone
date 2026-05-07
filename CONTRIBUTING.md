# Contributing

## Setup

```bash
git clone https://github.com/thebunnyweb/touchstone
cd touchstone
pip install -e ".[all]"
```

## Tests

```bash
pytest tests/
```

## Adding a provider

1. Create `touchstone/providers/yourprovider.py` implementing `BaseProvider`
2. Export it from `touchstone/providers/__init__.py`
3. Add auto-detection logic in `judge.py::_resolve_provider` if needed
4. Add docs page under `docs-site/pages/providers/`

## Adding a bias correction

1. Add detection logic to `touchstone/bias.py`
2. Wire it into `Judge.score()` or `Judge.compare()`
3. Add the flag name to the `bias_flags` list in the result
4. Document in `docs-site/pages/bias-correction.mdx`

## Submitting

- Keep PRs focused — one feature or fix per PR
- All public functions need type hints
- No new runtime dependencies without discussion
