# Tutorial: Automate DVC experiments (lesson 3)
## Machine Learning experiments reproducibility and engineering with DVC //  ML REPA School 

## 1. Clone this repository

```bash
git https://github.com/mlrepa/dvc-3-automate-experiments.git
cd dvc-3-automate-experiments
```

## 2. Create and activate virtual environment

Create virtual environment named `dvc` (you may use other name)
```bash
python3 -m venv dvc-venv
echo "export PYTHONPATH=$PWD" >> dvc-venv/bin/activate
source dvc-venv/bin/activate
```

## 3. Install python libraries

```bash
pip install -r requirements.txt
```

## 4. Add Virtual Environment to Jupyter Notebook

```bash
python -m ipykernel install --user --name=dvc-venv
``` 

## 5. Configure ToC for jupyter notebook (optional)

```bash
sudo jupyter contrib nbextension install
jupyter nbextension enable toc2/main
```

## 6. Run and follow Jupyter Notebook `dvc-3-automate-experiments.ipynb` for instructions:

```bash
jupyter notebook
```
