# Classifying Iris Species Using Logistic Regression

This project is just for me to understand more about logistic regression by following some tutorials available in the internet. The main website that I referred to is <a href='https://www.analyticsvidhya.com/blog/2018/05/24-ultimate-data-science-projects-to-boost-your-knowledge-and-skills/?'>Analytics Vidhya</a>

## Getting Started

This project uses the iris data set which can be downloaded <a href="https://archive.ics.uci.edu/ml/datasets/Iris">here</a> 

### Prerequisites

Import all the required libraries. I use the pandas' DataFrame to have a better understanding of the data set by looking at the descriptive statistics of the data set.

```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```
