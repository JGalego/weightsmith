# Run `just` to list recipes.

# List recipes
default:
    @just --list

# Install dependencies and the git hooks
setup: sync hooks

# Install dependencies from uv.lock
sync:
    uv sync

# Install the pre-commit hooks
hooks:
    uv run --with pre-commit pre-commit install

# Format the code
fmt:
    uv run ruff format .
    uv run ruff check --fix .

# Lint and check formatting without changing anything
lint:
    uv run ruff check .
    uv run ruff format --check .

# Type-check
typecheck:
    uv run pyright

# Run the tests; extra arguments go to pytest
test *args:
    uv run pytest {{ args }}

# Everything CI runs
check: lint typecheck test

# Run every pre-commit hook on every file
pre-commit:
    uv run --with pre-commit pre-commit run --all-files

# Build the synthetic models the examples load
fixtures *args:
    uv run weightsmith fixtures {{ args }}

# Statically check a program without running it
check-program program="examples/repair_mlp.wsm":
    uv run weightsmith check {{ program }}

# Run a program; extra arguments go to `weightsmith run`
run program="examples/repair_mlp.wsm" *args: fixtures
    uv run weightsmith run {{ program }} {{ args }}

# Render a program as a highlighted SVG or HTML snippet; extra arguments go to scripts/snippet.py
snippet source *args:
    uv run python scripts/snippet.py {{ source }} {{ args }}

# Regenerate the README's example image
readme-image:
    uv run python scripts/snippet.py assets/readme-example.wsm -o assets/readme-example.svg

# Package the VS Code extension and install it into the `code` on PATH
vscode:
    cd editors/vscode && npx --yes @vscode/vsce package --skip-license -o weightsmith.vsix
    code --install-extension editors/vscode/weightsmith.vsix --force

# Remove caches, build output and run output
clean:
    rm -rf .pytest_cache .ruff_cache dist build out
    find . -path ./.venv -prune -o -type d -name __pycache__ -exec rm -rf {} +
