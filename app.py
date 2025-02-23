import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("sc.pkl")  # Load the same scaler used in training

st.title("Titanic Prediction using Logistic Regression")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, encoding="utf-8")

    # Drop unwanted columns
    drop_cols = ["Cabin", "PassengerId", "Name", "Ticket"]
    data.drop(columns=[col for col in drop_cols if col in data.columns], inplace=True)

    # Handle missing values
    if "Embarked" in data.columns:
        data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)
    if "Age" in data.columns:
        data["Age"].fillna(data["Age"].mean(), inplace=True)
    if "Fare" in data.columns:
        data["Fare"].fillna(data["Fare"].median(), inplace=True)
    if "Pclass" in data.columns:  
        data["Pclass"].fillna(data["Pclass"].mode()[0], inplace=True)
    if "Sex" in data.columns:
        data["Sex"].fillna(data["Sex"].mode()[0], inplace=True)
    if "SibSp" in data.columns:
        data["SibSp"].fillna(data["SibSp"].mode()[0], inplace=True)
    if "Parch" in data.columns:
        data["Parch"].fillna(data["Parch"].mode()[0], inplace=True)



    le = LabelEncoder()
    if "Embarked" in data.columns:
        data["Embarked"] = le.fit_transform(data["Embarked"])
    if "Sex" in data.columns:
        data["Sex"] = le.fit_transform(data["Sex"])
    if "Sex" in data.columns:
        data["Pclass"] = le.fit_transform(data["Pclass"])
    if "Sex" in data.columns:
        data["SibSp"] = le.fit_transform(data["SibSp"])
    if "Sex" in data.columns:
        data["Parch"] = le.fit_transform(data["Parch"])




    # Ensure the order of columns matches training data
    feature_order = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    data = data[feature_order]

    # Check again for NaNs before transformation
    if data.isnull().sum().sum() > 0:
        st.error("Data still contains missing values. Please check your dataset.")
    else:
        # Scale the features using the same scaler from training
        data_scaled = scaler.transform(data)

        # Make predictions
        Survived = model.predict(data_scaled)
        survived = (Survived==1).sum()
        died = (Survived==0).sum()
        st.write(f"Predicted Survival: {(Survived==1).sum()}")
        st.write(f"Predicted Died: {(Survived==0).sum()}")

        fig , ax = plt.subplots()
        ax.pie([survived,died], labels=["survived", "died"], autopct="%1.1f%%", colors=["green","red"], startangle =90)
        ax.set_title("Survival")
        st.pyplot(fig)