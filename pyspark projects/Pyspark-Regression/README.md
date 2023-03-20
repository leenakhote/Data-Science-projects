# Basic Regression using PySpark ML Pipeline
## Overview
I have implemented a simple regression solution using the pyspark machine learning pipeline component. 

## Dataset
Medical Cost Personal Dataset from kaggle was used in this implementation. The dataset has the following features :

- age: age of primary beneficiary
- sex: insurance contractor gender, female, male
- bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
- children: Number of children covered by health insurance / Number of dependents
- smoker: Smoking
- region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
- charges: Individual medical costs billed by health insurance

Link to the dataset : https://www.kaggle.com/datasets/mirichoi0218/insurance


## High Level Design

<img width="869" alt="image" src="https://user-images.githubusercontent.com/89654615/205430816-939e891d-9597-4d84-9e1e-5e81819f309c.png">

1. The categorical features ('sex','smoker','region') are converted into ordinal variables using <code>pyspark.ml.feature.StringIndexer</code>
2. These feature are further transformed into one-hot encodings using <code>pyspark.ml.feature.OneHotEncoder</code>
3. All the features are combined into a single feature column, to be fed into the model (format expected by pyspark.ml) <code>pyspark.ml.feature.VectorAssembler</code>
4. Finally, we train a linear regression model with cross validation & hyper-parameter tuning <code>pyspark.ml.regression.LinearRegression</code>
5. These sequence of steps are encapsulated within a pipeline using <code>pyspark.ml.Pipeline</code>
