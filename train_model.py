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
df['species'] = y.map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Show dataset
st.subheader("Dataset Preview")
st.write(df.head())

# Descriptive stats
st.subheader("Descriptive Statistics")
st.write(df.describe())

# Visualizations
st.subheader("Feature Distributions by Species")
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(data=df, x='species', y='sepal length (cm)', ax=ax)
st.pyplot(fig)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Show accuracy
st.subheader("Model Performance")
st.write(f"Accuracy: {accuracy:.2f}")

# Predict new input
st.subheader("Predict New Sample")
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.4)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 3.8)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)[0]
species_names = ['setosa', 'versicolor', 'virginica']
st.write(f"Predicted Species: **{species_names[prediction]}**")
