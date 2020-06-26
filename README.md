# dvc-lesson-4

## 1. clone this repository

```bash
git clone https://gitlab.com/7labs.ru/tutorials-dvc/dvc-3-automate-experiments.git
cd dvc-lesson-4
```

## 2. Create and activate virtual environment

Install virtualenv in advance: 

```bash
pip install virtualenv
```

Create virtual environment 
```bash
virtualenv venv-dvc-3-automate-experiments
source venv-dvc-3-automate-experiments/bin/activate
```

## 3. Install python libraries (including dvc)

```bash
pip install -r requirements.txt
```

    
## 4. Add Virtual Environment to Jupyter Notebook

```bash
python -m ipykernel install --user --name=venv-dvc-3-automate-experiments
``` 

## 5. Run and follow Jupyter Notebook `Lesson 4.ipynb` for instructions:

```bash
jupyter notebook
```

