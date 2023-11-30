import pandas as pd
import tomli

from src.cltrier_promptClassify.dataset import Dataset

CONFIG: dict = tomli.load(open('./examples/config.toml', 'rb'))


def test_dataset():
    dataset = Dataset(**CONFIG['dataset'])

    assert type(dataset.df) is pd.DataFrame
    assert type(dataset.get_batches()[0]) is pd.DataFrame

    assert len(dataset.get_batches()[0]) == dataset.batch_size
    assert len(dataset.get_batches()) == len(dataset.df) // dataset.batch_size
