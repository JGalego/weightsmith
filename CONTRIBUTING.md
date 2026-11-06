# Contributing

```bash
uv sync                 # or `just setup`, which also installs the pre-commit hooks
uv run pytest           # unit and end-to-end tests; builds fixtures once into .pytest_cache
uv run ruff check . && uv run ruff format --check .
uv run pyright
```

`just check` runs everything CI runs. `just` lists the other recipes.

[docs/architecture.md](docs/architecture.md) maps the pipeline to the packages. [docs/semantics.md](docs/semantics.md) is the language definition; a test fails if a registry name is missing from it.

The README's images are generated: `just readme-image` renders `assets/readme-example.wsm`, and `just demo` re-records `assets/demo.gif` with [vhs](https://github.com/charmbracelet/vhs).
