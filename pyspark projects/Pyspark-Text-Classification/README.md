# Basic Text classification using PySpark ML Pipeline
## Overview
I have implemented a simple multi-class text classification solution using the pyspark machine learning pipeline component. 

## Dataset
Ecommerce text dataset from kaggle was used in this implementation. The textual column is a combination/concatenation of product title & product description

The dataset has the following features :

- Data Set Characteristics: Multivariate
- Number of Instances: 50425
- Number of classes: 4
- Number of Attributes: 1
- Attribute Characteristics: Text
- Associated Tasks: Multi-class Classification

Link to the dataset : https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification  


## High Level Design

<img width="869" alt="image" src="https://user-images.githubusercontent.com/89654615/205424415-433c4e8c-abf3-403b-8a09-a02058c067af.png">

1. We convert the raw text into tokens using the <code>pyspark.ml.feature.Tokenizer</code>
2. The tokenized text is transformed into feature vectors using the count vectorizer <code>pyspark.ml.feature.CountVectorizer</code>
3. The label is encoded to feed into the model using <code>pyspark.ml.feature.StringIndexer</code>
4. Finally, we train a logistic regression model with cross validation & hyper-parameter tuning, on the feature matrix <code>pyspark.ml.classification.LogisticRegression</code>
5. These sequence of steps are encapsulated within a pipeline using <code>pyspark.ml.Pipeline</code>
