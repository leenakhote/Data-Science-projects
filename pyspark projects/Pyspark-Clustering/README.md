# Basic Clustering using PySpark ML 
## Overview
I have implemented a simple Clustering solution using the pyspark machine learning library spark.ml 

## Dataset
Online Retail Data Set from kaggle was used from UCI ML repository. This is a transactional data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail

The dataset has the following features :

- InvoiceNo: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation.
- StockCode: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.
- Description: Product (item) name. Nominal.
- Quantity: The quantities of each product (item) per transaction. Numeric.
- InvoiceDate: Invice Date and time. Numeric, the day and time when each transaction was generated.
- UnitPrice: Unit price. Numeric, Product price per unit in sterling.
- CustomerID: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.
- Country: Country name. Nominal, the name of the country where each customer resides.

Link to the dataset : https://archive.ics.uci.edu/ml/datasets/online+retail#  


## High Level Design

<img width="945" alt="image" src="https://user-images.githubusercontent.com/89654615/205476532-c3873e21-079e-4c6f-933c-20a78a276555.png">
