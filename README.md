# Classifying Mushroom Edibility with Machine Learning

In this repository, we decribe and conduct a training process for various machine learning models. We are using UCI Mushroom dataset, which is availbale publicly: [UCI Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/Mushroom).

Objective: predict edibility of a mushroom, given categorical data.



## Enviroment

- Python 3.12 (global installation)
- Install requirements:

```setup
pip install -r requirements.txt
```
- Additionally, the course library **courselib** contains data-preprocessing tools and machine learning models that were used in this project. It can be found in this repository and must be installed/loaded before attempting the training.

## Contents

This project explores:

1) Different feature encodings: 
    - One-hot
    - Ordinal
    - Frequency.

2) Classification with:
    - Logistic Regression with L1 penalty (Lasso) for feature selection & sparsity 
    - Ridge Classifier (GD optimizer)
    - Linear and kernel SVMs.

## Training

The data-loading, data studies, as well as training processes for each model, can be found in the following jupyter notebook: [Mushroom.ipynb](Mushroom.ipynb)

## Results

Main accuracy results of the conducted training:

| Encoding   | Model                | Mean Accuracy |
|------------|----------------------|--------------:|
| Frequency  | Linear SVM           |         55.95 |
|            | Ridge classification |         92.93 |
| One-hot    | Linear SVM           |         98.02 |
|            | Ridge classification |         97.94 |
| Ordinal    | Linear SVM           |         59.07 |
|            | Ridge classification |         89.41 |

Additional outputs, such as sparsity graphs for each encoding method and hyperparameter selection for the kernel SVM method, can be found in the [Mushroom_Project_Report](ML_Project_Report.pdf), and as outputs of the Jupyter Notebook itself.
