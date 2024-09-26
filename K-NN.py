from math import sqrt
import sklearn
from sklearn.datasets import make_blobs
from matplotlib import pyplot
from pandas import DataFrame

#Function for calculating the distance between two points using the Euclidean distance
def calculate_distance(row1, row2):
    '''This function returns the distance between two points'''
    return sqrt(
        sum([(row1[0][i] - row2[0][i])**2 for i in range(len(row1[0]))])
        )
    
def get_neighbors(train, test_row, num_neighbors=3):
    #compute distances
    distances = [(train_row, calculate_distance(test_row, train_row))
    for train_row in train]
    #sort
    distances.sort(key=lambda tup: tup[1])
    #get top-k neighbors
    neighbors = [distances[i][0] for i in range(num_neighbors)]
    return neighbors

# Function to predict the class of a given test row
def predict_classification(train, test_row, num_neighbors=3):
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
X, y = make_blobs(n_samples=500, centers=3, n_features=2)

# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue', 2: 'green'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key,
color=colors[key])
pyplot.show()
print(sklearn.__version__)
