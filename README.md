# MLE_captsone_project


# Capstone Project: Create a Customer Segmentation Report for Arvato Financial Services
In this project, I analyzed demographics data for customers of a mail-order sales company in Germany, comparing it against demographics information for the general population. I used unsupervised learning techniques to perform customer segmentation, identifying the parts of the population that best describe the core customer base of the company. Then, I used supervised machine learning techniques to predict which individuals are most likely to convert into becoming customers for the company.  
I made a blog to shares my results. This is my blog post link: [blog post](https://oboukary.github.io/udacity_mle_nanodegree)

## Table of Contents
1) Project Motivation <br>
2) File Description <br>
3) Libraries Required <br>
4) Summary of Results <br>
5) Licensing and Acknowledgements <br>

## Project Motivation
To practice and showcase unsupervised and supervised machine learning skills and complete the Udacity machine learning Nanodegree capstone project.

## File Descriptions
**README.md** - This file, describing the contents of this repo

**.gitignore** - The gitignore file

**Arvato_capstone.ipynb** - Jupyter Notebook file containing project code

**DIAS Information Levels - Attributes 2017.xlsx** and **DIAS Attributes - Values 2017.xlsx** - Excel files containing information on the features.

**blog_images folder** - folder containing images used in my blog 

## Libraries Required
The code was written in python 3 and requires the following packages: 
```python
import math
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC,LinearSVC
import xgboost as xgb
from IPython.display import Image
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
```
## Installation
> Install xgboost by 

## Summary of Results
See GitHub repo and personnol [blog post](https://oboukary.github.io/udacity_mle_nanodegree) for summary of results.  

## Licenses and Acknowledgements
This project was completed as part of the Udacity machine learning Nanodegree.
Arvato provided the data which cannot be publicly shared.