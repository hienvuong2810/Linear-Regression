import pandas as pd

## read data from file csv with no header 
data =  pd.read_csv('data.csv',header= None)

## X is data input, Y is data output
## Example: X[0] = 1 -> Y[0] = 2
##	    X[1] = 2 -> Y[1] = 4	
## 3->6, 4->8, 5->10, etc...

X = data.values[:,0]
Y = data.values[:,1]


## model_predict is find predict the output value of parameter X, weight and bias found by define 'trainning' 
def model_predict(number,weight,bias):
	return weight*number + bias
## loss_function is MSE (measures squared error), compute average of MSE to to increase accuracy
## Recipe MSE : 
def loss_function(X,Y, weight, bias):
	n = len(X)
	sum_loss = 0
	for i in range(n):
		sum_loss += (Y[i] - (weight * X[i] + bias)) **2
	
	return sum_loss/n
## update_weight_and_bias optimization weight and bias 
def update_weight_and_bias (X,Y,weight, bias, learning_rate):
	n = len(X)
	weight_temp = 0.0
	bias_temp = 0.0
	for i in range(n):
		weight_temp += -2 * X[i]*(Y[i] - (X[i]* weight + bias))
		bias_temp   += -2*(Y[i] - (X[i] * weight + bias))
	weight -= (weight_temp/n) * learning_rate
	bias   -= (bias_temp/n)   * learning_rate

	return weight, bias
## trainning is function to train 
def trainning (X, Y, weight, bias, learning_rate, iter):
	loss_history = []
	for i in range(iter):
		weight, bias = update_weight_and_bias(X, Y, weight, bias, learning_rate)
		error = loss_function(X, Y, weight, bias)
		loss_history.append(error)
	return weight, bias
weight, bias = trainning (X, Y, 0.00001, 0.00001, 0.001, 10000)
## You can change variable 'number_want_to_predict' to know which number it will predict 
number_want_to_predict = 32
print('Predict: ', model_predict(number_want_to_predict, weight, bias))



