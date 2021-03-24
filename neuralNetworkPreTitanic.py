import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# import matplotlib.pyplot as plt
# import seaborn as sns
import os


# functions
# function that return the first letter of Cabin
def desk(string):
    prima = string[0]
    return prima


# function that return if a passenger has a cabin or not
def if_cabin(string):
    if string == 'M':
        return 0
    else:
        return 1


# function that determine if a name is a long name
def over_40(string):
    if len(string) > 40:
        return 1
    else:
        return 0


# checking relating information
for dirname, _, filenames in os.walk("E:/Pycharm/WestTwo/Python_FourthTerm/Data"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train = pd.read_csv("E:/Pycharm/WestTwo/Python_FourthTerm/Data/train.csv")
print("--info about train data--")
train.info()
print(train.isnull().sum())
print(train.isnull().mean())
print(train.nunique())
# describe the rate of data skew
print(train.describe())
# describe the relations between data
print(train.corr())

test = pd.read_csv("E:/Pycharm/WestTwo/Python_FourthTerm/Data/test.csv")
print("\n--info about test data--")
test.info()
print(test.isnull().sum())
print(test.isnull().mean())
print(test.nunique())
# describe the rate of data skew
print(test.describe())

# process the data

# fill some blank column
# Cabin
train.Cabin = train.Cabin.fillna("Missing")
test.Cabin = test.Cabin.fillna("Missing")
train["deck"] = train["Cabin"].apply(desk)
test["deck"] = test["Cabin"].apply(desk)
# train Age
train["Age"] = train["Age"].fillna(train["Age"].median())
# train Embark
train["Embarked"] = train["Embarked"].fillna("S")
# test Age
test["Age"] = test["Age"].fillna(test["Age"].median())
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

# add name length attribute
train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)

# add log_Fare attribute(due to the high skew rate of fare)
train["log_fare"] = np.log1p(train["Fare"])
test["log_fare"] = np.log1p(test["Fare"])

# add attribute about a passenger has a cabin
train["if_cabin"] = train.deck.apply(if_cabin)
test["if_cabin"] = test.deck.apply(if_cabin)

data = [train, test]
for dataset in data:
    dataset["relatives"] = dataset["SibSp"] + dataset["Parch"]
    dataset.loc[dataset["relatives"] > 0, 'not_alone'] = 0
    dataset.loc[dataset["relatives"] == 0, 'not_alone'] = 1
    dataset["not_alone"] = dataset["not_alone"].astype(int)

# add attribute about a passenger's name length over 40
train["over40"] = train["Name"].apply(over_40)
test["over40"] = test["Name"].apply(over_40)

print(train.isnull().mean())
print(test.isnull().mean())

# Train the model
features = ["Sex", "log_fare", "Age", "if_cabin", "relatives", "not_alone", "Name_length", "over40"]
x = pd.get_dummies(train[features])
x_test = pd.get_dummies(test[features])
y = train["Survived"]

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
model.fit(x, y)
predictions = model.predict(x_test)
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved")
