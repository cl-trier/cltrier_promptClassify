import logging
from typing import List

from src.cltrier_promptClassify.integrations import HuggingFace


def test_huggingface():
    model = HuggingFace('bigscience/mt0-small')
    inputs: List[str] = ['This is the first test.', 'This is the second test']

    generated = model.inference(inputs)
    logging.debug(generated)

    assert type(generated) is list
    assert type(generated[0]) is str
    assert len(generated) == len(inputs)
