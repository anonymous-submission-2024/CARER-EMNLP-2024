
This is the official repository for the EMNLP 2024 submission: **CARER** - **C**linic**A**l **R**easoning-**E**nhanced **R**epresentation for Temporal Health Risk Prediction. \
The source code will be refactored and published soon!

## Download the MIMIC-III and MIMIC-IV datasets
Go to [https://mimic.physionet.org/](https://mimic.physionet.org/gettingstarted/access/) for access. Once you have the authority for the dataset, download the dataset and extract the csv files to `data/mimic3/raw/` and `data/mimic4/raw/` in this project.

## Preprocess ICD code and notes
```bash
python run_preprocess.py
```

## Preprocess Lab values and generate verbalized queries
```bash
python generate_queries.py
```

## Generate clinical reasoning
```bash
python generate_reasoning.py
```


## Train model
```bash
python train.py
```

