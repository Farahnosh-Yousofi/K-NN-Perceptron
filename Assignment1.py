import pandas as pd

'''
Q1. Load the dataset using Pandas:
Write a python script to load the dataset into a Pandas DataFrame. Assign appropriate
column names: [‘sepal_length’, ‘sepal_width’, ‘petal_length’, ‘petal_width’, ‘species’].

'''

file_path = "iris/iris.data"
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

# Load the dataset into a DataFrame
iris_df = pd.read_csv(file_path, names=column_names)

# Display the first few rows of the DataFrame
print(iris_df.head())

'''
Question 2:
What are the dimensions of the dataset? Use Pandas to find the number of rows and columns
'''

# Get the number of rows and columns
num_rows = iris_df.shape[0]
num_cols = iris_df.shape[1]

print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_cols}")
