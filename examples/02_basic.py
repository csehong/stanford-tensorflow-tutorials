import tensorflow as tf
a = tf.constant(2, name ='a')
b = tf.constant(3, name = 'b')
x = tf.add(a, b, name = 'x')
writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
with tf.Session() as sess:
	# writer = tf.summary.FileWriter('./graphs', sess.graph)
	print(sess.run(x))
writer.close()


con_zero = tf.zeros([2, 3], tf.int32)
con_one = tf.ones([2, 3], tf.int32)

with tf.Session() as sess:
    print (sess.run(con_zero))
    print (sess.run(con_one))


norm = tf.random_normal([2, 3], mean=-1, stddev=4)
c = tf.constant([[1, 2], [3, 4], [5, 6]])
shuff = tf.random_shuffle(c)
unif = tf.random_uniform([2,3], minval=0, maxval=3)

print ("Random\n")
with tf.Session() as sess:
    print(sess.run(norm))
    print(sess.run(shuff))
    print(sess.run(unif))


x = tf.constant([0.5, 2.5, 2.3, 1.5, 3.5, 4.5, -3.5, -4.5])
x_round = tf.round(x)
with tf.Session() as sess:
    print(sess.run(x_round))


a = tf.constant([2, 2], name='a')
b = tf.constant([[0, 1], [2, 3]], name='b')
c = tf.constant([2.0, 2.0], name='c')
d = tf.constant([[0.0, 1.0], [2.0, 3.0]], name='d')
with tf.Session() as sess:
    print(sess.run(tf.div(b, a)))
    print(sess.run(tf.divide(b, a)))
    print(sess.run(tf.div(d, c)))
    print(sess.run(tf.divide(d, c)))

W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
    sess.run(W.initializer)
    print(W.eval())
    sess.run(assign_op)
    print(W.eval())


# create a variable whose original value is 2
my_var = tf.Variable(2, name="my_var")

# assign a * 2 to a and call that op a_times_two
my_var_times_two = my_var.assign(2 * my_var)

with tf.Session() as sess:
    sess.run(my_var.initializer)
    print(my_var.eval())
    sess.run(my_var_times_two) 				# >> the value of my_var now is 4
    print(my_var.eval())
    sess.run(my_var_times_two) 				# >> the value of my_var now is 8
    print(my_var.eval())
    sess.run(my_var_times_two) 				# >> the value of my_var now is 16
    print(my_var.eval())



a = tf.placeholder(tf.float32, shape=[3])

# create a constant of type float 32-bit, shape is a vector of 3 elements
b = tf.constant([5, 5, 5], tf.float32)

# use the placeholder as you would a constant or a variable
c = a + b  # Short for tf.add(a, b)

with tf.Session() as sess:
	print(sess.run(c, {a: [1, 2, 3]}))



sess = tf.InteractiveSession()
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
print(c.eval()) # we can use 'c.eval()' without explicitly stating a session
sess.close()
