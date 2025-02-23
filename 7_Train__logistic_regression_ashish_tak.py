import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import io
import joblib



dftr = pd.read_csv("Titanic_train.csv")

""" 2. Preprocessing handling missing values and drop unwanted columns"""
dftr.drop(columns = ["Cabin","PassengerId","Name","Ticket"], inplace= True)
dftr.dropna(subset=['Embarked'], inplace = True)
dftr["Age"] = dftr["Age"].fillna(dftr["Age"].mean())


"""Label encoding"""
le = LabelEncoder()
dftr["Embarked"] = le.fit_transform(dftr["Embarked"])
dftr["Sex"] = le.fit_transform(dftr["Sex"])


"""Model Building"""
x = dftr.drop(columns = ["Survived"])
y = dftr["Survived"]

print(x)

sc = StandardScaler()
x = sc.fit_transform(x)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state = 42)

model = LogisticRegression(class_weight='balanced')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)


joblib.dump(model, "model.pkl")
joblib.dump(sc, "sc.pkl")  


print(f"Accuracy Score:    {accuracy_score(y_test, y_pred):.4%}")
print(f"Precision Score:   {precision_score(y_test, y_pred):.4%}")
print(f"Recall Score:      {recall_score(y_test, y_pred):.4%}")
print(f"F1 Score:          {f1_score(y_test, y_pred):.4%}")
