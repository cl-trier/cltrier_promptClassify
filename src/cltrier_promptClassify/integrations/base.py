from abc import ABC, abstractmethod
from typing import List

from tqdm import tqdm

from ..dataset import Dataset


class BaseIntegration(ABC):

    def __call__(self, prompt: str, dataset: Dataset) -> List[str]:
        predictions: List[str] = []

        for batch in tqdm(dataset.get_batches(), leave=True):
            predictions.extend(
                self.inference(
                    [prompt.replace('{text}', sample)
                     for sample in batch[dataset.text_column].values]
                )
            )

        return predictions

    @abstractmethod
    def inference(self, batch: List[str], **kwargs) -> List[str]:
        pass
