test:
	@python3 -m pytest tests/

# ---

example:
	@python3 -m src.cltrier_promptClassify ./examples/config.toml

# ---

deploy:
	flit publish
