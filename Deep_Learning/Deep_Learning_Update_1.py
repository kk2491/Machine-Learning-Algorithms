# import tensorflow as tf 
import keras
print(keras.__version__)
mnist = keras.datasets.mnist   

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = keras.utils.normalize(x_train, axis = 1)
x_test = keras.utils.normalize(x_test, axis = 1)

'''
import matplotlib.pyplot as plt 
plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()
print(x_train[0])
'''

# 2 Types of model
# 1. Sequential - Like feedforward
# 2. ???

model = keras.models.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation = nn.relu))    # number of neurons = 128
model.add(keras.layers.Dense(128, activation = nn.relu))	  
model.add(keras.layers.Dense(10, activation = nn.softmax))

# loss - degree of error
# adam - optimizer
# crossentrory - how it works ? 
model.compile(optimizer = 'adam',
			  loss = 'sparse_categorical_crossentropy',
			  metrics = ['accuracy'])

# How about batch size ?
model.fit(x_train, y_train, epochs = 3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

model.save_weights('/home/kk/Github/Machine-Learning-Algorithms/Deep_Learning/epic_num_reader.model')

new_model = keras.models.load_model('/home/kk/Github/Machine-Learning-Algorithms/Deep_Learning/epic_num_reader.model')

predictions = new_model.predict([x_test])

print(predictions)






