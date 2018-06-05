import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import backend
import pickle
import os
#import clusterone

def to_float_list(l):
	"""
	[to_float_list] takes a list of string representations of
		float and converts that to a list of floats.
	"""
	temp_list = []
	for i in l:
		try:
			temp_list.append(float(i))
		except:
			temp_list.append(0.)
	return temp_list

def convert_data(data):
	"""
	[convert_data] takes in raw data from a csv file and cleans it up so
		that it can be used in a NN.
	Return format: x_data, y_data
		where both are numpy arrays of lists
	[data] - data taken from a csv file
	"""
	# indices in the data where values are already float and not options
	# (Read: they do not need to be represented as a one-hot vector)
	value_indices = [0,3,4,17,18,19,20,26,34,36,37,38,43,44,45,46,47,48,49,
					 50,51,52,54,56,59,61,62,66,67,68,69,70,71,75,76,77,80]
	
	# indices which represent years (for more useful normalization practices)
	years = [19,20,59,77]
	
	# creating a list of categories, so fields with contain options can be
	# accurately represented as a one-hot vector
	length = len(data[0])
	categories = []
	for i in range(length):
		categories.append([])
	for vector in data:
		for i, field in enumerate(vector):
			if field not in categories[i]:
				categories[i].append(field)
				
	for i in range(length):
		categories[i].sort()
	
	# using the category representation above to transform the data into
	# usable vectors
	x_data = []
	y_data = []
	for vector in data:
		new_vector = []
		for i, field in enumerate(vector):
			# if field is an option, create one-hot vector and extend the list
			if (i not in value_indices):
				index = categories[i].index(field)
				cat_len = len(categories[i])
				temp_list = []
				for m in range(cat_len):
					temp_list.append(0.)
				temp_list[index] = 1.
				new_vector.extend(temp_list)
			# if field is already a float, append it to the list
			else:
				max_val = max(to_float_list(categories[i]))
				# if field is a year, first subtract 1900, and then normalize and append
				if i in years:
					try:
						new_vector.append((float(field)-1900)/(max_val-1900))
					except:
						new_vector.append(0.)
				else:
					try:
						new_vector.append(float(field)/(max_val))
					except:
						new_vector.append(0.)
		
		# separate the input and output data
		x_data.append(new_vector[1:-1])
		y_data.append([new_vector[-1]])
		
	# formatting so the numpy array has type 'float'
	h,w = len(x_data),len(x_data[0])
	
	x = np.zeros((h,w),dtype=float)
	y = np.zeros((h,1),dtype=float)
	
	for i in range(h):
		x[i,:] = x_data[i]
		y[i,:] = y_data[i]
	
	return x, y
				

def get_data(train = 1000, val = 300):
	"""
	[get_data] returns training data that has been properly formatted
		for the purpose of running it through a NN.
	Return format: x_train, x_val, x_test, y_train, y_val, y_test
		All of which are numpy arrays.
	[train] - number of training samples
	[val] - number of validation samples
		(test samples will be whatever is left)
	"""
	# opening up and retrieving training data
	data = []
	with open("train.csv",'r') as F:
		temp_data = F.readlines()
		for i,field in enumerate(temp_data):
			if i>0: # discard first line (labels)
				d = field.replace("\n","")
				data.append(d.split(","))
	
	# turning the data into a useful form
	x_data, y_data = convert_data(data)
	
	# separating data into training, validation, and testing
	x_train = x_data[:train]
	x_val = x_data[train:train+val]
	x_test = x_data[train+val:]
	y_train = y_data[:train]
	y_val = y_data[train:train+val]
	y_test = y_data[train+val:]
	
	return x_train, x_val, x_test, y_train, y_val, y_test


if __name__ == "__main__":
	# turn of warnings
	os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

	# root mean squared metric
	def rmse(y_true, y_pred):
		return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
	
	print("Begin collecting data")
	x_train, x_val, x_test, y_train, y_val, y_test = get_data()
	print("Finished collecting data")

	input_nodes = x_train.shape[1]

	# layers: (input: 326), 1000, 500, 200, 100, (output: 1)
	# dropout of 10% between each layer
	model = Sequential()
	model.add(Dense(1000,input_dim=input_nodes,activation="relu"))
	model.add(Dropout(0.1))
	model.add(Dense(500,activation="relu"))
	model.add(Dropout(0.1))
	model.add(Dense(200,activation="relu"))
	model.add(Dropout(0.1))
	model.add(Dense(100,activation="relu"))
	model.add(Dropout(0.1))
	model.add(Dense(1,activation="tanh"))
	
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=[rmse])
	
	# model.load_weights("test3.h5")
	
	# running the model
	history = model.fit(x_train,y_train,epochs=30,batch_size=25,validation_data=(x_val,y_val))
	model.save_weights("test5.h5")

	with open('logs/run_a', 'wb') as file_pi:
		pickle.dump(history.history, file_pi)

	# evaluating test data
	score = model.evaluate(x_test,y_test)
	print('Test loss:', score[0])
	print('Test error:', score[1])
	
	# retrieving stats on accuracy by using the model to predict house values
	max_cost = 755000.0
	cumulative_error = 0.
	samples = 160
	for i in range(samples):
		x = x_test[i]
		y = y_test[i]
		
		actual = y[0]*max_cost
		prediction = (model.predict(np.array([x]))*max_cost)[0][0]
		error = (prediction-actual)/actual*100.
		cumulative_error += abs(error)/float(samples)
		
		print("Actual: ",actual, ", Prediction: ",prediction, " Percent Error: ", error,"%")

	print("Cumulative Error: ", cumulative_error,"%")
	