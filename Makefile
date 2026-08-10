.PHONY: test lint data forecast backtest figures ablation router report all

test:
	uv run pytest

lint:
	uv run ruff check src tests && uv run ruff format --check src tests

data:
	uv run w25 download

forecast: data
	uv run w25 forecast

backtest:
	uv run w25 backtest --stress

figures:
	uv run w25 figures-data && uv run w25 figures-results && uv run w25 ablation && uv run w25 figures-misc

router:
	uv run w25 router

report:
	mkdir -p report/figures
	cp 01_tech_watch/figures/*.png 02_backtest_harness/figures/*.png 03_complexity_router/figures/*.png \
	   04_reference_reading/figures/*.png report/figures/
	cd report && QUARTO_PYTHON=../.venv/bin/python quarto render

all: forecast backtest figures router report
