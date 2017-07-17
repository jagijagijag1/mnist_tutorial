from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import input_data

import tensorflow as tf


# Load MNIST Data
print("data loading...")
mnist = input_data.read_data_sets("MNIST_DATA", one_hot=True)
print("DONE")

# Start TensorFlow InteractiveSession
sess = tf.InteractiveSession()

# Placeholders
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# Variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Initialize variables (in this case tensors full of zeros)
sess.run(tf.global_variables_initializer())

# Predicted Class (regression model)
y = tf.matmul(x, W) + b

# Loss Function
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Train the Model (steepest gradient descent)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

print("training...")
for _ in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
print("DONE")

# Evaluate the Model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("The result (accuracy) is")
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
