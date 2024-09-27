
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Question 3
'''
Iris Dataset: This dataset contains 150 instances of Iris flowers,
which has 3 classes each class has a different type of Iris flower with respect to their features,
sepal length in cm, sepal width in cm, petal length in cm, petal width in cm.
This dataset is used to evaluating classification methods, and the best classification method for this dataset is Neural Network classification,
because its  mean accuracy is higher (dot further right) and has shorter error bars (less variability) based on the baseline performance model figures.

'''


# Question 4
# Step 1: Load the iris dataset into a pandas DataFrame
iris_file_path = "iris/iris.data"  # Update this path
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_df = pd.read_csv(iris_file_path, header=None, names=column_names)

# Step 2: Separate features and labels
X = iris_df.drop('class', axis=1).values  # Features
y = iris_df['class'].values               # Labels

# Step 3: Split the data
# Split into training set (60%) and test set (40%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Split temp set into validation set (20%) and test set (20%)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Display the shapes of the splits to verify
print(f'Training set: {X_train.shape}, {y_train.shape}')
print(f'Validation set: {X_val.shape}, {y_val.shape}')
print(f'Test set: {X_test.shape}, {y_test.shape}')



def calculate_distance(row1, row2, distance_metric='euclidean'):
    if distance_metric == 'euclidean':
        return np.sqrt(np.sum((row1 - row2) ** 2))
    elif distance_metric == 'manhattan':
        return np.sum(np.abs(row1 - row2))
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

def get_neighbors(train, test_row, num_neighbors, distance_metric='euclidean'):
    distances = [(train_row, calculate_distance(test_row, train_row, distance_metric))
                 for train_row in train]
    distances.sort(key=lambda tup: tup[1])
    neighbors = [distances[i][0] for i in range(num_neighbors)]
    return neighbors

def predict_classification(train, train_labels, test_row, num_neighbors, distance_metric='euclidean'):
    neighbors = get_neighbors(train, test_row, num_neighbors, distance_metric)
    neighbor_labels = [train_labels[np.where((train == neighbor).all(axis=1))[0][0]] for neighbor in neighbors]
    prediction = max(set(neighbor_labels), key=neighbor_labels.count)
    return prediction

def find_best_k(X_train, y_train, X_val, y_val, distance_metric='euclidean'):
    k_values = range(1, 21)
    best_k = 1
    best_score = 0
    for k in k_values:
        predictions = [predict_classification(X_train, y_train, row, k, distance_metric) for row in X_val]
        accuracy = accuracy_score(y_val, predictions)
        if accuracy > best_score:
            best_score = accuracy
            best_k = k
    return best_k, best_score


#Question 5: Evaluate with Different Distance Metrics
def evaluate_knn(X_train, y_train, X_test, y_test, num_neighbors, distance_metric='euclidean'):
    predictions = [predict_classification(X_train, y_train, row, num_neighbors, distance_metric) for row in X_test]
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')
    return accuracy, precision, recall, f1

# Evaluating Euclidean and Manhattan distances
best_k_euclidean, _ = find_best_k(X_train, y_train, X_val, y_val, distance_metric='euclidean')
euclidean_results = evaluate_knn(X_train, y_train, X_test, y_test, best_k_euclidean, distance_metric='euclidean')

best_k_manhattan, _ = find_best_k(X_train, y_train, X_val, y_val, distance_metric='manhattan')
manhattan_results = evaluate_knn(X_train, y_train, X_test, y_test, best_k_manhattan, distance_metric='manhattan')

print("Best k Euclidean distances:", best_k_euclidean)
print("Euclidean Results:", euclidean_results)

print("Best k Manhattan distances:", best_k_manhattan)
print("Manhattan Results:", manhattan_results)


