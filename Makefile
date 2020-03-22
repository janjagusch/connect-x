clean:
	@echo "Cleaning up ..."
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +

test_missing_init:
	@echo "Testing for missing __init__.py ..."
	@poetry run python bin/test_missing_init

test_tox: test_missing_init
	@echo "Tox testing ..."
	@poetry run tox

test: test_tox clean

format_black: test_missing_init
	@echo "Black formatting ..."
	@poetry run black .

format: format_black clean

lint_black: test_missing_init
	@echo "Black linting ..."
	@poetry run black --check .

lint_pylint: test_missing_init
	@echo "Pylint linting ..."
	@poetry run pylint connect_x
	@poetry run pylint $$(find tests/ -iname "*.py")

lint: lint_black lint_pylint clean

