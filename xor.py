import tensorflow as tf
import numpy as np

def inference(training, hidden_units):
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.random_uniform([2, hidden_units], 0.0, 1.0), name='weights')
        biases = tf.Variable(tf.zeros([hidden_units]), name='biases')
        hidden1 = tf.sigmoid(tf.matmul(training, weights) + biases)
    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.random_uniform([hidden_units, 1], 0.0, 1.0), name='weights')
        biases = tf.Variable(tf.zeros([1]), name='biases')
        hidden2 = tf.sigmoid(tf.matmul(hidden1, weights) + biases)
    return hidden2

batch_size = 4
inputs_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 2))
expected_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 1))
actual = inference(inputs_placeholder, 12)

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(actual - expected_placeholder))
optimizer = tf.train.GradientDescentOptimizer(0.7)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

inputs = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
expected = np.array([[0.0], [0.0], [1.0], [1.0]])

iterations = 0
for step in xrange(20000):
    _, loss_value = sess.run([train, loss], feed_dict={
        inputs_placeholder: inputs,
        expected_placeholder: expected})
    iterations = iterations + 1
    if loss_value <= 0.00075:
        break

print iterations
print sess.run(actual, feed_dict={inputs_placeholder: inputs})