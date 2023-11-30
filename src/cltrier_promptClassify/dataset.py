from typing import List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict


class Dataset(BaseModel):
    text_column: str
    gold_column: str = None

    df: Optional[pd.DataFrame] = None
    path: Optional[str] = None

    batch_size: Optional[int] = 32

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **obj):
        super().__init__(**obj)

        if not self.df:
            if not self.path:
                raise ValueError('Could not load data. Not path provided')

            else:
                self.__load_df()

    def __load_df(self):
        fl_path = Path(self.path)

        if not fl_path.is_file():
            raise ValueError(f'Could not load data. Invalid path: {self.path}')

        else:
            self.df = pd.read_csv(self.path)

    def get_batches(self) -> List[pd.DataFrame]:
        return list(zip(*(self.df.groupby(np.arange(len(self.df)) // self.batch_size))))[1]
