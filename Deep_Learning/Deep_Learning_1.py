import tensorflow as tf 

tf.__version__
print(tf.__version__)

x1 = tf.constant(5)
x2 = tf.constant(6)

result = tf.multiply(x1, x2)
# result = x1 * x2 

print(result)

'''
sess = tf.Session()
print(sess.run(result))
sess.close()
'''

# Best practice

with tf.Session() as sess:
	output = sess.run(result)
	print(output)

print(output)
