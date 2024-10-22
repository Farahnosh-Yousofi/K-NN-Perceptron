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

'''
Question 3:
Display the first 10 rows of the dataset to familiarize yourself with the data structure.
'''

# Display the first 10 rows of the DataFrame
print(iris_df.head(10))

"""
    Question 4:
Write Pandas script to find the following:
• The number of unique species in the dataset.
• The count of each species.
"""
num_unique_species = iris_df['species'].nunique()
species_counts = iris_df['species'].value_counts()

print(f"Number of unique species: {num_unique_species}")
print(f"Count of each species:\n{species_counts}")

'''
Question 5:
Calculate the mean, median, and standard deviation for each numerical column (sepal length,
sepal width, petal length, petal width).
'''

for feature in column_names:
    if feature in ['sepal_length','sepal_width', 'petal_length', 'petal_width']:
    # Calculate the mean, median, and standard deviation for each feature
        mean = iris_df[feature].mean()
        median = iris_df[feature].median()
        std_dev = iris_df[feature].std()

        print(f"Statistics for {feature}:")
        print(f"Mean: {mean}")
        print(f"Median: {median}")
        print(f"Standard Deviation: {std_dev}")
        print("\n")
        
'''
Question 6:
Generate a 5x5 NumPy array filled with random numbers between 0 and 1. Perform the
following operations on the array:
• Calculate the sum of all elements.
• Find the maximum and minimum values.
• Compute the mean for each row and each column.
'''

import numpy as np

# Generate a 5x5 NumPy array filled with random numbers between 0 and 1
random_array = np.random.rand(5, 5)

# Calculate the sum of all elements
array_sum = np.sum(random_array)

# Find the maximum and minimum values
max_value = np.max(random_array)
min_value = np.min(random_array)

# Compute the mean for each row and each column
row_means = np.mean(random_array, axis=1)
column_means = np.mean(random_array, axis=0)

print(f"Sum of all elements: {array_sum}")
print(f"Maximum value: {max_value}")
print(f"Minimum value: {min_value}")
print(f"Mean of each row:\n{row_means}")
print(f"Mean of each column:\n{column_means}")

'''
Question 7:
• Create a 3x3 identity matrix using NumPy.
• Create another 3x3 matrix with numbers from 1 to 9.
Multiply the identity matrix by the 3x3 matrix and display the result.
'''

# Create a 3x3 identity matrix using NumPy
identity_matrix = np.eye(3)
print(identity_matrix)

# Create another 3x3 matrix with numbers from 1 to 9
matrix_9 = np.arange(1, 10).reshape(3, 3)
print(matrix_9)

# Multiply the identity matrix by the 3x3 matrix
result = np.dot(identity_matrix, matrix_9)
print(f"Result of multiplying identity matrix with 3x3 matrix:\n{result}")
