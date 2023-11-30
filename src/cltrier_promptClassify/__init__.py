"""
Description needed
"""

__version__ = "0.0.1"

import logging
from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd

from .dataset import Dataset
from .integrations import HuggingFace
from .util import setup_logging


class Pipeline:

    def __init__(self, config: dict):
        setup_logging()
        self.config: dict = config
        self.dataset: Dataset = Dataset(**self.config['dataset'])

    def __call__(self) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
        results: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {}

        logging.info(f'> classification')
        if self.config['do_classification']:
            results['classify'] = Pipeline.classify(data=self.dataset, **self.config['classify'])

        if self.config['do_evaluation']:
            logging.info(f'> evaluation')
            results['eval'] = Pipeline.evaluate(results['classify'], self.dataset.gold_column, 'prediction')

        if self.config['do_export']:
            logging.info(f'> export')
            for result_label, result_data in results.items():
                Pipeline.export(result_data, self.config['export_path'], result_label)

        return results

    @staticmethod
    def classify(
            models: List[Tuple[str, str]],
            templates: List[Tuple[str, str]],
            classes: List[str],
            data: Dataset
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        results: Dict[str, Dict[str, pd.DataFrame]] = {}

        for model_label, model_slug in models:
            integration = HuggingFace(model_slug)

            results[model_label] = {
                templates_label: data.df.assign(prediction=integration(
                    template_content.replace('{classes}', str(classes)),
                    data
                ))
                for templates_label, template_content in templates
            }

        return results

    @staticmethod
    def evaluate(data: Dict[str, Dict[str, pd.DataFrame]], *args):
        return {
            model_label: {
                template_label: Pipeline.__evaluate_df(template_df, *args)
                for template_label, template_df in model_data.items()
            }
            for model_label, model_data in data.items()
        }

    @staticmethod
    def __evaluate_df(
            df: pd.DataFrame,
            gold_col: str,
            pred_col: str,
    ):
        return (
            df
            .pipe(lambda _df: _df.assign(metric=df[gold_col].str.lower() == df[pred_col].str.lower()))
            .groupby(gold_col)['metric']
            .value_counts(normalize=True, dropna=False)
            .unstack()
            .pipe(lambda _df: 1.0 - _df[False])
            .fillna(0.0)
        )

    @staticmethod
    def export(data: Dict[str, Dict[str, pd.DataFrame]], path: str, prefix: str = None):
        path = Path(path)
        path.mkdir(exist_ok=True)

        for model_label, model_data in data.items():
            for template_label, template_df in model_data.items():
                template_df.to_csv(Path(path).joinpath(f'{prefix}.{model_label}.{template_label}.csv'))
