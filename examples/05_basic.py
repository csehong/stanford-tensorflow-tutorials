import tensorflow as tf

x = tf.Variable(2.0)
y = 2.0 * (x ** 3)
z = 3.0 + y ** 2
grad_z = tf.gradients(z, [x, y])
with tf.Session() as sess:
	sess.run(x.initializer)
	print(sess.run(grad_z)) # >> [768.0, 32.0]
# 768 is the gradient of z with respect to x, 32 with respect to y
