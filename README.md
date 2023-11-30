# CLTrier PromptClassify

## Usage

### Python Module

```python
from cltrier_promptClassify import Pipeline

# init pipeline object
pipeline = Pipeline({
    # pipeline objects
    'do_classification': True,
    'do_evaluation': True,
    'do_export': True,
    # path to export dir (only if do_export)
    'export_path': './path/dir/',
    # dataset configuration
    'dataset': {
        # path to data file (.csv)
        'path': './path/file.csv',
        # column containing src text
        'text_column': 'text',
        # column containing gold label (only if do_evaluation)
        'gold_column': 'gold',
        # (optional) batch size used during classification
        'batch_size': 16,
    },
    # classifier configuration
    'classifiers': [
        # label for export, slug/url from hugging face hub
        ['model_label', 'model_huggingface_slug'],
        # ...
    ],
    'templates': [
        # {classes}, {text} are dynamically replaced during runtime
        ['template_label', 'prompt_template (must include {classes} and {text})'],
        # ...
    ],
    # list of classes to use
    'classes': ['class_1', 'class_2']
})

# call pipeline object
pipeline()
```

### Terminal Script

```bash
python3 -m cltrier_promptClassify ./path/to/config.toml
```

```toml
# pipeline objects
do_classification = true
do_evaluation = true
do_export = true
# path to export dir
export_path = './path/dir/'

# dataset configuration
[dataset]
# path to data file (.csv)
path = './path/file.csv'
# column containing src text
text_column = 'text'
# column containing gold label (only if do_evaluation)
gold_column = 'gold'
# (optional) batch size used during classification
batch_size = 16

# classifier configuration
[classify]
# label for export, slug/url from hugging face hub
models = [
    ['model_label', 'model_huggingface_slug'],
    # ...
]
# {classes}, {text} are dynamically replaced during runtime
templates = [
    ['template_label', 'prompt_template (must include {classes} and {text})'],
]
# list of classes to use
classes = ['class_1', 'class_2']
```
