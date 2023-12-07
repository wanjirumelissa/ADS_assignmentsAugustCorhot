#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
iris_data = pd.read_csv(url, header=None, names=columns)

# Display the raw data
st.title("Iris Dataset Explorer")

# Display the raw data
st.subheader("Raw Data")
st.write(iris_data)

# Question 1: Show the average sepal length for each species
st.subheader("Average Sepal Length by Species")
average_sepal_length = iris_data.groupby("species")["sepal_length"].mean()
st.bar_chart(average_sepal_length)

# Question 2: Display a scatter plot comparing two features
st.subheader("Scatter Plot")
feature1 = st.selectbox("Select the first feature", iris_data.columns[:-1])
feature2 = st.selectbox("Select the second feature", iris_data.columns[:-1])
scatter_plot = sns.scatterplot(data=iris_data, x=feature1, y=feature2, hue="species")
st.pyplot(scatter_plot.get_figure())

# Question 3: Filter data based on species
st.subheader("Filter Data by Species")
selected_species = st.selectbox("Select a species", iris_data["species"].unique())
filtered_data = iris_data[iris_data["species"] == selected_species]
st.write(filtered_data)

# Question 4: Display a pairplot for the selected species
st.subheader("Pairplot for Selected Species")
pairplot = sns.pairplot(filtered_data, hue="species")
st.pyplot(pairplot)

# Question 5: Show the distribution of a selected feature
st.subheader("Distribution of Selected Feature")
selected_feature = st.selectbox("Select a feature", iris_data.columns[:-1])
distribution_plot = sns.histplot(data=iris_data, x=selected_feature, hue="species", kde=True)
st.pyplot(distribution_plot.get_figure())

# Save the Streamlit app
if st.button("Save Streamlit App"):
    st.code("streamlit run data_explorer.py")

# Save the seaborn plots to files
scatter_plot.get_figure().savefig("scatter_plot.png")
pairplot.savefig("pairplot.png")
distribution_plot.get_figure().savefig("distribution_plot.png")


# In[ ]:




