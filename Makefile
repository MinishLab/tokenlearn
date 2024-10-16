venv:
	uv venv

install: venv
	uv sync --all-extras
	uv run pre-commit install

fix:
	uv run pre-commit run --all-files

test:
	uv run pytest --cov=nanofit --cov-report=term-missing
