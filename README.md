# Risky Borrowers Classification with PyTorch

![Python](https://img.shields.io/badge/python-3.10+-orange.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.7.1+-red.svg)
![Jupyter](https://img.shields.io/badge/jupyter-1.1.1+-blue.svg)
![NumPy](https://img.shields.io/badge/numpy-2.2.6+-lightgrey.svg)
![Pandas](https://img.shields.io/badge/pandas-2.3.1+-yellow.svg)
![PySpark](https://img.shields.io/badge/pyspark-3.4.0-orange.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.0+-brightgreen.svg)
[![License: MIT](https://img.shields.io/badge/license-MIT-red.svg)](https://opensource.org/licenses/MIT)
<br>

<p align="center">
  <img src="images/A-simple-neural-network-diagram-with-one-hidden-layer.png" alt="Title Image">
</p>
<br>

<p align="center">
  <a href="#summary"> Background and Introduction </a> •
  <a href="#data"> Sourcing data </a> •
  <a href="#process"> Data Transformation and EDA</a> •
  <a href="#eda"> Visualisation </a> •
  <a href="#models"> Model </a> •
  <a href="#eval"> Evaluation </a> •
  <a href="#conc"> Conclusions</a>
</p>

<a id = 'summary'></a>

## Background and Introduction

This project is a port to PyTorch from a collaborated project I did with some colleagues. The task proposed selection of a loan/credit default dataset and develop classification models to predict or investigate factors that influence whether a borrower would default, recorded as `0` (non-default) or `1` (default) in the `default_ind` variable.

The neural networks in Tensorflow, but I saw PyTorch offered a better interface and development experimence, so the port was made.

In this exercise, I investigated a variation of the LoanStatNew dataset using PyTorch. I performed dimensionality reduction, features selection and class weighting to compare the performance of different model architectures.

See the notebooks here: [Analysis Notebook](./src/data-processing_visualisation.ipynb) and [Training Notebook](./src/preprocessing_model-fitting.ipynb)

<a id = 'data'></a>

## Sourcing data

The dataset is a variation of Kaggle's Lending Club dataset. with addition features that were created based on the original dataset. It offers a wide view of different aspects in the borrowers' financial status, and a high flexibility for model selection.

The [data dictionary with schema definiton](./data_dict.md) can be found in the `.md` file.

<a id = 'process'></a>

## Data Transformation

The proposed processing pipeline includes the following steps:

- Dropping a range of columns that were deemed uninformative or have high proportion of missing values

- Dropping rows (sice data size was adequate) for columns that have low missingness, including the target, `default_ind`

- Transforming the time-based columns into Timestamp type, stored as number of days since **1970-01-01**

- Preprocess remaining categorical or numerical variables using MLlib's `StringIndexer()` estimator and casting respectively

- Used median to fill any missing or 'Unknown' values after acquiring the **train set**

Overall, the dataset's quality is relatively high, no inconsistent string values that require `regex` were found. The only other prominent issue is class imbalance, which was alleviated using class weights.

| default_ind | count |
|-----------|------|
|0 |647004 |
|1 | 37149 |

Addtionally, a design change I made that's different to the original work was to store processed data in `.parquet` format instead of `.csv`. In the project, code was written in a single notebook so we didn't have issues with replication as we used the `.toPandas()` method to convert dataset into tensors via numpy.

However, I'd like to improve modularity, so I modified the pipeline.

<a id = 'eda'></a>

## Visualisation

<p align="center">
   <img src="images/heatmap.png" alt="Correlation Heatmap"  width="500">
<p/>

<p align="center">
   <img src="images/histogram.png" alt="Distribution of Relevant Features"  width="500">
<p/>

After visualising the underlying patterns of the dataset, we employed `compute_class_weight()` from scikit-learn to calculate the bias for model training.

<a id = 'models'></a>

## Building the models

Using `nn.Module`, I created 3 classes, a Multilayer Perceptron (MLP), a bagging model with MLP base models and a convolutional network (CNN) with 1-D convolution layers. The models' architecture are relatively simple as I initially built them using Keras' documentation.

The first two model has some Dense/Linear layers, while the CNN has the usual design. All models are validated using a `DataLoader` instance crated from the validation set.

```python
val_loader = DataLoader(val_tensor_ds, 
                        batch_size=32, 
                        shuffle=False, 
                        pin_memory=True if device.type == 'cuda' else False)
```

<a id = 'eval'></a>

## Evaluation

- The MLP and bagging model achieved high performance, with precision and recall ~99%.

  - The inclusion of class weights in the loss function may have reduced the selection bias within these models and may have improved the results.

- The CNN model performed noticeably worse to the other two, which is expected CNNs are not typically fit on tabular data.

- The area under the curve (AUC) are close to 1 across all networks, meaning they are very likely to make the correct classification.

  - This high AUC result also suggests potential overfitting, but regularlisation/normalisation was applied, so this suggests presence of leakage.

Below is the summary of the Tensorflow models' performance on the **test set**.

|Model | Accuracy| AUC | Precision | Recall | Loss | False Negative Count |
|---------------|--------|--------|--------|--------|--------|--------|
| Multilayer Perceptron | 0.9981 | 0.9992 | 0.9763 | 0.9884 | 0.0313 | 107.0 |
| Ensemble MLP |0.9958| 0.9984 | 0.9378 | 0.9868 | 0.0506 | 122.0 |
| 1D Convolutional Network | 0.9651 | 0.9651 | 0.7551 | 0.5254 | 0.1647 | 4394.0|

For PyTorch:

```text

====================== Multilayer Perceptron ======================
Test Accuracy: 0.9964450142782213
Precision: 0.9685643296964459
Recall: 0.9656435110393107
ROC AUC: 0.9906845145536228

====================== Bagging with MLP base ======================
Test Accuracy: 0.9971210443499039
Precision: 0.9612761045230349
Recall: 0.9865374259558427
ROC AUC: 0.9987235500711122

====================== CNN with 1-D Convoluitonal layers ======================
Test Accuracy: 0.9458884550381724
Precision: 0.0
Recall: 0.0
ROC AUC: 0.9009592831143862
```

The convolutional network was not able to classify the other label during test, possible causing 0 precision and recall despite having informative AUC and Accuracy.

<a id = 'conc'></a>

## Conclusion

We've prototyped tree-based models previously, which were more robust and achieved better performance compared to neural networks. On the other hand, PyTorch's interface offers a nice development experience and is easier to scale with its CUDA integration.
