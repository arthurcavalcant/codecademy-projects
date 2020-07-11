import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
passengers = pd.read_csv("passengers.csv")
print(passengers)

# Update sex column to numerical
passengers["Sex"] = passengers["Sex"].map({"female": 1, "male": 0})

# Fill the nan values in the age column
meanAge = passengers["Age"].mean()
passengers["Age"].fillna(value = meanAge, inplace = True)

# Create a first class column
passengers["FirstClass"] = np.where(passengers["Pclass"] == 1, 1, 0)

# Create a second class column
passengers["SecondClass"] = np.where(passengers["Pclass"] == 2, 1, 0)

# Select the desired features
features = passengers[["Sex", "Age", "FirstClass", "SecondClass"]]
survival = passengers["Survived"]

# Perform train, test, split
x_train, x_test, y_train, y_test = train_test_split(features, survival, test_size = 0.2)

# Scale the feature data so it has mean = 0 and standard deviation = 1
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Create and train the model
model = LogisticRegression()
model.fit(x_train, y_train)

# Score the model on the train data
print(model.score(x_train, y_train))

# Score the model on the test data
print(model.score(x_test, y_test))

# Analyze the coefficients
print(dict(zip(['Sex','Age','FirstClass','SecondClass'], model.coef_[0])))

# Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
Me = np.array([0.0,18.0,0.0,1.0])

# Combine passenger arrays
sample_passengers = np.array([Jack, Rose, Me])

# Scale the sample passenger features
sample_passengers = scaler.transform(sample_passengers)
print(sample_passengers)

# Make survival predictions!
print(model.predict(sample_passengers))