import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Title 
st.title("Iris Species Classifier")
st.write("A simple data analytics and ML app")

# Load data 
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')
df = X.copy()
df['species'] = y.map({0: 'setosa', 1: 'versicolor', 2: 'verginica'})

# Show dataset
st.subheader("Dataset preview")
st.write(df.head())

# Visualizations
st.subheader("Feature Distributions by species")
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(data=df, x='species', y='sepal length (cm)', ax=ax)
st.pyplot(fig)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model performance")
st.write(f"Accuracy: {accuracy:.2f}")

st.subheader("Predict new sample")
sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.4)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal length (cm)" 1.0, 7.0, 3.8)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.2)

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)[0]
species_names = ['setosa', 'versicolor', 'virginica']
st.write(f"Predicted species: **{species_names[prediction]}**")