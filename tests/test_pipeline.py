import tomli

from src.cltrier_promptClassify import Pipeline

CONFIG: dict = tomli.load(open('./examples/config.toml', 'rb'))


def test_pipeline():
    pipeline = Pipeline(CONFIG)
    results = pipeline()

    assert type(results) is dict
    assert type(results.items()) is dict
