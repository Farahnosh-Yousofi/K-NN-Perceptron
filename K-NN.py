from math import sqrt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib import pyplot
import pandas as pd
from pandas import DataFrame

#Function for calculating the distance between two points using the Euclidean distance
def calculate_distance(row1, row2):
    '''This function returns the distance between two points'''
    return sqrt(
        sum([(row1[0][i] - row2[0][i])**2 for i in range(len(row1[0]))])
        )
    
def get_neighbors(train, test_row, num_neighbors=6):
    #compute distances
    distances = [(train_row, calculate_distance(test_row, train_row))
    for train_row in train]
    #sort
    distances.sort(key=lambda tup: tup[1])
    #get top-k neighbors
    neighbors = [distances[i][0] for i in range(num_neighbors)]
    return neighbors

# Function to predict the class of a given test row
def predict_classification(train, test_row, num_neighbors=6):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    # get the most frequent class
    class_votes = {}
    for neighbor in neighbors:
        response = neighbor[-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.items(), key=lambda item: item[1], reverse=True)
    return sorted_votes[0][0]

# generate 2d classification dataset
# Fixed data points (X) with 2 features each
X = [
    [1.0, 2.0],  # Point 1
    [2.5, 3.5],  # Point 2
    [1.5, 1.8],  # Point 3
    [8.0, 8.0],  # Point 4
    [9.0, 9.5],  # Point 5
    [7.5, 8.0],  # Point 6
    [2.0, 8.0],  # Point 7
    [1.5, 8.5],  # Point 8
    [8.0, 2.0],  # Point 9
    [7.5, 1.5]   # Point 10
]
X = np.array(X)
# Corresponding labels (y) for each point in X
y = [
    0,  # Label for Point 1
    0,  # Label for Point 2
    0,  # Label for Point 3
    1,  # Label for Point 4
    1,  # Label for Point 5
    1,  # Label for Point 6
    2,  # Label for Point 7
    2,  # Label for Point 8
    3,  # Label for Point 9
    3   # Label for Point 10
]

# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue', 2: 'green', 3: 'yellow'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key,
color=colors[key])
pyplot.show()

X= X.tolist()
dataset = [(data, label) for data, label in zip(X,y)]
test_data_point = ( [1, 2], 1)
prediction = predict_classification(dataset, test_data_point)

print(prediction)


#Creating confusion matrix
actual_data = pd.Series([1,2,3,1,3,3,2,1,1,1], name = "Actual")
predicted_data = pd.Series([1,2,1,2,1,3,3,2,1,1], name = "Predicted")
df_confusion_matrix = pd.crosstab(actual_data, predicted_data, rownames=['Actual'], colnames=['Predicted'], margins=True)
print(df_confusion_matrix)

#Calculating accuracy
accuracy = (df_confusion_matrix.iloc[0,0] + df_confusion_matrix.iloc[1,1] + df_confusion_matrix.iloc[2,2]) / df_confusion_matrix.iloc[3,3]
print(f'Accuracy: {accuracy}')


# Calculate precision, recall, and F1-score for each class
# Setting average=None returns metrics for each class
precision = precision_score(actual_data, predicted_data, average=None)
recall = recall_score(actual_data, predicted_data, average=None)
f1 = f1_score(actual_data, predicted_data, average=None)

# Display precision, recall, and F1-score for each class
classes = sorted(actual_data.unique())
for i, cls in enumerate(classes):
    print(f"\nClass {cls}:")
    print(f"Precision: {precision[i]:.2f}")
    print(f"Recall: {recall[i]:.2f}")
    print(f"F1-Score: {f1[i]:.2f}")

