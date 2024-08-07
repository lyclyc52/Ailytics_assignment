

## Installation

```bash
conda create -n ailytics_env python=3.10 -y
conda activate ailytics_env
conda install -c conda-forge poppler # install this to process pdf
pip install --upgrade pip  # Enable PEP 660 support.
pip install -e ".[inference]"
```