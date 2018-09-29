#This is to predict whether the credit card customer will be default in the coming month

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split

data = pd.read_csv("Credit_Card_Data.csv")
data.shape

data.head()

del data["ID"]

ind_data = data.iloc[:,0:23]
dep_data = data.iloc[:,23] 

#First you divide the data into train and test
train_x, test_x, train_y, test_y = train_test_split(ind_data, dep_data, train_size = 0.80, random_state = 3)

scalar = Normalizer()
scalar.fit(train_x)
train_x = scalar.transform(train_x)

train_x = pd.DataFrame(train_x)
train_y = pd.DataFrame(train_y)

test_x = scalar.transform(test_x)
test_x = pd.DataFrame(test_x)
test_y = pd.DataFrame(test_y)

#Lets start the tensorflow computational graph from here 
n_nodes_hl1 = 5
n_nodes_hl2 = 5
n_nodes_hl3 = 5

n_classes = 1
batch_size = 100

x = tf.placeholder('float',[None, 23])
y = tf.placeholder('float')

def neural_network_model(data):
    
    hidden_1_layer = {'weights' : tf.Variable(tf.random_normal([23, n_nodes_hl1])),
                      'biases' : tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases' : tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases' : tf.Variable(tf.random_normal([n_nodes_hl3]))}
    
    output_layer = {'weights' : tf.Variable(tf.random_normal([n_nodes_hl1, n_classes])),
                    'biases' : tf.Variable(tf.random_normal([n_classes]))}
    
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    
    output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']
    #print (output)    
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    #prediction = np.array(prediction)
    #print(prediction)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = prediction, labels = y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    #hm_epochs = 10
    
    hm_epochs = 20
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        for epoch in range(hm_epochs):
            
            epoch_loss = 0
            i = 0
            length = len(train_x)
            #print("Length ",length)
            while i < length:
                #print("Length of train is ",
                start = i
                end = i + batch_size
                
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])            
                #print(batch_x[1:5])
                #print(batch_y[1:5])
                #print("Length of train :",len(batch_x),"length of y :",len(batch_y))
                _, c = sess.run([optimizer, cost], feed_dict = {x : batch_x, y : batch_y})
                
                #print(c)
                epoch_loss += c
                i = i + batch_size
                
            print("Epoch ", epoch, "completed out of ", hm_epochs, ". Loss is : ",epoch_loss)
        
        #Currently working on finding a better evaluation method as argmax is not working for binary classification
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print("Accuracy of training set :",accuracy.eval({x:train_x, y:train_y}))
        
        print("Accuracy of testing set :",accuracy.eval({x:test_x, y:test_y}))
        
train_neural_network(x)

