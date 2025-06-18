import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly_express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.title("EGG PRODUCTION ANALYSIS")
st.markdown("## OVERVIEW")

#import my csv file
st.markdown("### FIRST TEN OBSERVATIONS")
df = pd.read_csv("egg_production_system.csv")
st.write(df.head(10))

st.markdown("### LAST TEN OBSERVATIONS")
df = pd.read_csv("egg_production_system.csv")
st.write(df.tail(10))

st.markdown("### DATA INFO")
AK = df.shape
st.write(AK)


st.markdown("### Number of eggs from hens in non-organic, free-range farms")
st.write(df["Number of eggs from hens in non-organic, free-range farms"].describe())

st.markdown("### Number of eggs from hens in non-organic, free-range farms")
st.write(df["Number of eggs from hens in non-organic, free-range farms"].head(10))


#UNIVARIATE ANALYSIS
st.markdown("# UNIVARIATE ANALYSIS")
st.markdown("### EGG NUMBERS ANALYSIS")
df = pd.read_csv("egg_production_system.csv")
st.write(df["Number of eggs from hens in non-organic, free-range farms"].describe())

st.markdown("### YEAR ANALYSIS")
df = pd.read_csv("egg_production_system.csv")
st.write(df["Year"].describe())

st.markdown("### EGGS INORGANIC FREE RANGE ANALYSIS")
df = pd.read_csv("egg_production_system.csv")
st.write(df["Number of eggs from hens in organic, free-range farms"].describe())

st.markdown("### EGGS IN BARNS ANALYSIS")
df = pd.read_csv("egg_production_system.csv")
st.write(df["Number of eggs from hens in barns"].describe())

st.markdown("### EGGS IN ENRICHED CAGE ANALYSIS")
df = pd.read_csv("egg_production_system.csv")
st.write(df["Number of eggs from hens in (enriched) cages"].describe())

st.markdown("### CODE ANALYSIS")
df = pd.read_csv("egg_production_system.csv")
st.write(df["Code"].describe())

"""
st.markdown("### HISTOGRAM REPRESENTATION FOR BP")
BP = px.histogram(df["BloodPressure"], y= "BloodPressure", title="Pressure Distribution")
st.plotly_chart(BP, use_container_width=True)

st.markdown("### LINE GRAPH REPRESENTATION FOR BP")
BP = px.line(df["BloodPressure"], y= "BloodPressure", title="Pressure Distribution")
st.plotly_chart(BP, use_container_width=True)

st.markdown("### BAR REPRESENTATION FOR BP")
BP2 = px.bar(df["BloodPressure"], y= "BloodPressure", title="Pressure Distribution")
st.plotly_chart(BP2, use_container_width=True)

st.markdown("### HISTOGRAM REPRESENTATION FOR PREGNANCIES")
Pregg = px.histogram(df["Pregnancies"], y ="Pregnancies", title = "Pregnancies Distribution")
st.plotly_chart(Pregg, use_container_width = True)

st.markdown("### LINE GRAPH REPRESENTATION FOR PREGNANCIES")
Pregg = px.line(df["Pregnancies"], y ="Pregnancies", title = "Pregnancies Distribution")
st.plotly_chart(Pregg, use_container_width = True)

st.markdown("### BAR REPRESENTATION FOR PREGNANCIES")
Pregg = px.bar(df["Pregnancies"], y ="Pregnancies", title = "Pregnancies Distribution")
st.plotly_chart(Pregg, use_container_width = True)

#BIVARIATE ANALYSIS
st.markdown("## BIVARIATE ANALYSIS")
st.markdown("### Blood Pressure vs Pregnancies")
df2 = pd.DataFrame(df["BloodPressure"],df["Pregnancies"])
st.write(df2)

st.markdown("### Blood Pressure vs BMI")
df3 = pd.DataFrame(df["BloodPressure"],df["BMI"])
st.write(df3)

st.markdown("### Glucose vs Pregnancies")
df4 = pd.DataFrame(df["Glucose"],df["Pregnancies"])
st.write(df4)

st.markdown("### Skin Thickness vs Pregnancies")
df5 = pd.DataFrame(df["SkinThickness"],df["Pregnancies"])
st.write(df5)

st.markdown("### Age vs Pregnancies")
df6 = pd.DataFrame(df["Age"],df["Pregnancies"])
st.write(df6)

st.markdown("### Pregnancies vs Insulin")
df_ = pd.DataFrame(df["Pregnancies"],df["Insulin"])
st.write(df_)

st.markdown("# PREDICTIVE ANALYSIS")
X = df.drop("Outcome", axis=1)
Y = df["Outcome"]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

model = LogisticRegression()
model.fit(X_train,Y_train) #training the model

st.markdown("## Outcome Prediction")
prediction = model.predict(X_test)
st.write(prediction)

st.markdown("## Model Evaluation")
accuracy = accuracy_score(prediction, Y_test)
st.write(accuracy)
"""
#download by typing "python -m pip install scikit-learn"
#download by typing "python -m pip install matlib"
#download by typing "python -m pip install seaborn"
#download by typing "python -m pip freeze > requirements.txt"
