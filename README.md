# AI-Powered Drug Discovery & Quantum Protein Folding

A drug-discovery proof-of-concept: featurizes SMILES strings, trains classical ML, deep-learning and quantum/hybrid models (PennyLane), and exposes a Streamlit UI to explore results.

## Contents
* `app.py` — Streamlit application (UI)
* `data_loader.py` — SMILES featurization using RDKit
* `ml_model.py` — classical ML training (RandomForest)
* `dl_model.py` — simple neural network (TensorFlow / Keras)
* `qml_model.py` / `qnn_model.py` — PennyLane quantum / hybrid models
* `test_all_models.py` — basic unit tests
* `mols_xx.jpg` — images used by the app

## Quick start

### Option A — (Conda, recommended for RDKit)
1. Create and activate environment:
```bash
conda create -n projecttt python=3.9 -y
conda activate projecttt
conda install -c conda-forge rdkit pandas numpy scikit-learn matplotlib seaborn pennylane -y
# Install remaining packages via pip
pip install -r requirements.txt
```

### Option B — (pip only; may have trouble with RDKit)
```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Run the Streamlit app
```bash
streamlit run app.py
```

### Run tests
```bash
pytest -q
```

## Dataset
The dataset used in development is not uploaded here and was sourced from Kaggle: [Big Molecules SMILES Dataset](https://www.kaggle.com/datasets/yanmaksi/big-molecules-smiles-dataset).

## License
Currently no license selected (all rights reserved).
