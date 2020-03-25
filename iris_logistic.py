'''
Analysis on Iris Dataset
using logistic regression
'''

# Import required libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

'''
Load Iris data
Set the header to none since the data set does not have the header
Substitute with a list of names
'''
iris = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
    header=None,
    names=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
)

# Trying to understand the data:
print(iris.head(),'\n')
print(iris.tail())
print('\n\nColumn Names:')
[print(a) for a in iris.columns]
print('\nNumber of rows:\n',iris.shape[0],'\nNumber of columns:\n',iris.shape[1],'\n')

# Descriptive statistics of the data
print(iris.describe(include='all'))

'''
Encode label the target variable
Label encoder encode labels with a value between 0 and n_classes-1
This is a part of preprocessing
We call it as dummy variable
'''

# For comparison, we copy a dataframe:
iris1 = iris.copy()

# Create a label encoder object
label_encoder = LabelEncoder()
print(type(label_encoder))

# Encode labels in column 'species'
print('Labels before encoded')
[print(b) for b in iris['Species'].unique()]
iris['Species'] = label_encoder.fit_transform(iris['Species'])
print(iris['Species'].unique())
print(iris.head())

# Another way is:
label_encoder1 = LabelEncoder()
new = label_encoder1.fit(iris1['Species'])
iris1['Species'] = new.transform(iris1['Species'])
print(iris1['Species'].unique())
print(iris1.head())


# Checking whether the dataframes are the same or not
def check_similarity(col1, col2):
    i = 0
    for index in range(len(col1)):
        if col1[index] == col2[index]:
            i += 1
    return i / len(col1) * 100


print(check_similarity(iris1['Species'], iris['Species']))
print(check_similarity(iris1['Species'], iris['SepalWidthCm']))

# Split the data set into train and test
# Train and test is used as verification
train, test = train_test_split(iris, test_size=0.2, random_state=0)
print(train.head(),'\nPercentage split:',train.shape[0]/iris.shape[0]*100,'%\n')
print(test.head(),'\nPercentage split:',test.shape[0]/iris.shape[0]*100,'%\n')

# Separate the independent and dependent variable
# Our dependent variable is the column 'Species'
# Train set:
# axis = 1 means that target column will be drop from data frame
train_x = train.drop(columns=['Species'], axis=1)
print(train_x.head(2))

# axis = 0 means that target index/row will be drop from data frame
# train_x1 = train.drop([137], axis=0)
# print(train_x1.head(2))
train_y = train['Species']

# Test set:
test_x = test.drop(columns=['Species'])
test_y = test['Species']

# Create the object of the model we want:
model = LogisticRegression(max_iter=150)

# Use the training set - both train_x and train_y:
model.fit(train_x,train_y)

# Predict test_x data set to produce prediction
prediction = model.predict(test_x)
[print(q) for q in prediction]
print('\nPredicted values on test data:')
[print(w) for w in label_encoder.inverse_transform(prediction)]

# print(len(prediction))

# Accuracy score on test data
print('\nAccuracy Score: ',accuracy_score(test_y,prediction),'\n')

# Let us understand a little bit more about the prediction:
# Look at some of the stat, and we will use it in the model we created
# The first species:
spec0 = iris.loc[iris['Species'] == 0,:'PetalWidthCm']
print('Sentosa:\n',spec0.mode())

# The second species:
spec1 = iris.loc[iris['Species'] == 1,:'PetalWidthCm']
print('Versicolor:\n',spec1.mode())

# The third species:
spec2 = iris.loc[iris['Species'] == 2,:'PetalWidthCm']
print('Virginica:\n',spec2.mode())


print('Enter sepal length in cm:')
SepalLength = input('> ')

print('Enter sepal length in cm:')
SepalWidth = input('> ')

print('Enter petal length in cm:')
PetalLength = input('> ')

print('Enter petal width in cm:')
PetalWidth = input('> ')

pred = pd.DataFrame(
    [
        {
            'SepalLengthCm': float(SepalLength),
            'SepalWidthCm': float(SepalWidth),
            'PetalLengthCm': float(PetalLength),
            'PetalWidthCm': float(PetalWidth)
        }
    ]
)
print(pred)
print(model.predict(pred))
print('Prediction:\n',label_encoder.inverse_transform(model.predict(pred))[0])
