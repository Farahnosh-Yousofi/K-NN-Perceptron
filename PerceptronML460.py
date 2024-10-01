
# Defining the function predict
def predict(rows, weights):
    '''Returns the list of dot products between elements of rows and weights'''
    
    return (rows[i]*weights[i] for i in range(len(weights)))


#Testing the function predict
weights = [1.0, 1.0, -2.0]  #list of weights

#The dataset
dataset = [
([1.0,2.0,1.0], 1), ([0.0,1.0,1.0], -1),
([2.0,1.0,1.0], 1), ([1.0,0.0,1.0], -1)
]

#Applying the function predict
for row in dataset:
    prediction  =  sum(predict(row[0], weights))*row[1]
    print("Expected=%d, Prediction=%s" % (row[1], 
                                          "Correct" if prediction >= 0
                                          else "Wrong"))



# Weights update function
# w <- w + yx
def weight_update(row, weights, labels):
    '''Updates the weights based on the given row, labels and weights'''
    return  [(weights[i]+ labels*row[i]) for i in range(len(weights))]


# Estimate the perceptron weights using peceptron
def train_weights(train, n_epoch):
    """Train weights for the given number of epochs"""
    weights = [0.0 for i in range(len(train[0][0]))]
    print(weights)
    for epoch in range(n_epoch):
        m = 0
        for row in train:
            prediction = sum(predict(row[0], weights))*row[1]
            if prediction <= 0:
                weights = weight_update(row[0], weights, row[1])
                print(weights)
                m += 1
        print("Epoch %d, Misclassified samples=%d" % (epoch, m))
        if m == 0:
            break
    print(weights)
    return weights

n_epoch = 5
weights = train_weights(dataset, n_epoch)
print(weights)