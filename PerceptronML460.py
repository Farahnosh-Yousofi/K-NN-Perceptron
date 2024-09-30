
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

