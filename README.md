# Classifying Iris Species Using Logistic Regression

This project is just for me to understand more about logistic regression by following some tutorials available in the internet. The main website that I referred to is <a href='https://www.analyticsvidhya.com/blog/2018/05/24-ultimate-data-science-projects-to-boost-your-knowledge-and-skills/?' target="_blank">Analytics Vidhya</a>

## Getting Started

This project uses the iris data set which can be downloaded <a href="https://archive.ics.uci.edu/ml/datasets/Iris" target="_blank">here</a> 

### Prerequisites

Import all the required libraries. I use the pandas' DataFrame to have a better understanding of the data set by looking at the descriptive statistics of the data set. Later, I use the LabelEncoder to change the three classes of the species to a dummy variable such that 0 represents Iris-Sentosa, 1 represents Iris-Versicolor, and 2 represents Iris-Virginica. The train_test_split is used to separate the data into two parts. One part is the train set, and the orher is the test set. Logistic regression model is trained by the train set and later verified for the accuracy score using the test set.

```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```
In the last part of the coding, this simple program will ask for input from the user. For better understanding, I use the mode value of each column for each species. For instance, majority of flowers from Iris-Virginica species have the following value:
<table>
  <tr>
    <th>SepalLengthCm</th>
    <th>SepalWidthCm</th>
    <th>PetalLengthCm</th>
    <PetalWidthCm</th>
  </tr>
  <tr>
    <td>6.3</td>
    <td>3.0</td>
    <td>5.1</td>
    <td>1.8</td>
  </tr>
</table>
If we take these values as input, then it is expected that the algorithm will return Iris-Virginica as the prediction.

## Built With

* [Pandas](https://pandas.pydata.org/)
* [Sci-kit Learn](https://scikit-learn.org/stable/)


## Acknowledgments

* Thanks to Analytics Vidhya for sharing so many useful information and tutorial which can be easily followed by a novice like me.
