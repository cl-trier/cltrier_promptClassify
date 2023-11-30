from typing import List

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .base import BaseIntegration
from ..util import get_device


class HuggingFace(BaseIntegration):

    def __init__(self, slug: str):
        self.tokenizer = AutoTokenizer.from_pretrained(slug)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(slug).to(get_device())

    def inference(
            self,
            batch: List[str],
            max_input_len: int = 512,
            max_output_len: int = 24
    ) -> List[str]:
        inputs: dict = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest",
            max_length=max_input_len,
            truncation=True,
        ).to(get_device())

        outputs = self.model.generate(**inputs, max_new_tokens=max_output_len)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
