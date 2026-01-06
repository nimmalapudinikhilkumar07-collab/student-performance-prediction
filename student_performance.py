import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
print("PREDICTING THE STUDENT FINAL SCORE USING AI MODEL")
df=pd.read_csv("student_data.csv")
df.isnull().sum()
df.drop_duplicates()
print(df.describe())

# whether check for the correlation then proceed to train model
print("correlation between them is")
print(df.corr())
X=df[["study_hours","attendance","previous_score"]]
y=df["final_score"]

# now training and testing and predicting starts

model=LinearRegression()
X_test,X_train,y_test,y_train=train_test_split(X,y,test_size=0.2,random_state=43)

model.fit(X_train,y_train)
y_predict=model.predict(X_test)

print("Actual values:",y_test.values)
print("Predicted values:",y_predict)

# we can check the performance using this

print(model.score(X_train,y_train))
print(model.score(X_test,y_test))

# calculate the mse and r2 square score to check model how good it was

mse=mean_squared_error(y_test,y_predict)
r2=r2_score(y_test,y_predict)
print("mean squared is :",mse)
print("r2 score is:",r2)

# now labelling the regression line to using graphs

plt.scatter(df["study_hours"],df["final_score"])
plt.xlabel("study_hours")
plt.ylabel("final_score")
plt.title("student final marks prediction")
plt.show()
plt.grid()