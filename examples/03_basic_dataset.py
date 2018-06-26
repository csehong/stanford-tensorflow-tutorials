import tensorflow as tf
import numpy as np

x = np.random.sample((100,2))
# make a dataset from a numpy array
dataset = tf.data.Dataset.from_tensor_slices(x)
# create the iterator
iter = dataset.make_one_shot_iterator()
el = iter.get_next()
with tf.Session() as sess:
    print(sess.run(el))
    print(sess.run(el))
    print(sess.run(el))
    print(sess.run(el))
    print(sess.run(el))



features, labels = (np.random.sample((100,2)), np.random.sample((100,1)))
dataset2 = tf.data.Dataset.from_tensor_slices((features,labels))
# create the iterator
iter = dataset2.make_one_shot_iterator()
el = iter.get_next()
with tf.Session() as sess:
    print(sess.run(el))
    print(sess.run(el))
    print(sess.run(el))
    print(sess.run(el))
    print(sess.run(el))




#****************************************************************************************


print("\n\n")
# using a placeholder
x = tf.placeholder(tf.float32, shape=[None,2])
dataset = tf.data.Dataset.from_tensor_slices(x)
data = np.random.sample((100,2))
data2 = np.random.sample((50,2))
iter = dataset.make_initializable_iterator() # create the iterator
el = iter.get_next()
with tf.Session() as sess:
    # feed the placeholder with data
    sess.run(iter.initializer, feed_dict={ x: data })
    print(sess.run(el)) # output [ 0.52374458  0.71968478]
    print(sess.run(el))  # output [ 0.52374458  0.71968478]
    print(sess.run(el))  # output [ 0.52374458  0.71968478]
    print(sess.run(el))  # output [ 0.52374458  0.71968478]
    print(sess.run(el))  # output [ 0.52374458  0.71968478]

    sess.run(iter.initializer, feed_dict={x: data2})
    print(sess.run(el))  # output [ 0.52374458  0.71968478]
    print(sess.run(el))  # output [ 0.52374458  0.71968478]
    print(sess.run(el))  # output [ 0.52374458  0.71968478]
    print(sess.run(el))  # output [ 0.52374458  0.71968478]
    print(sess.run(el))  # output [ 0.52374458  0.71968478]



print("\n\n")
# initializable iterator to switch between dataset
EPOCHS = 10
x, y = tf.placeholder(tf.float32, shape=[None,2]), tf.placeholder(tf.float32, shape=[None,1])
dataset = tf.data.Dataset.from_tensor_slices((x, y))
train_data = (np.random.sample((100,2)), np.random.sample((100,1)))
test_data = (np.array([[1,2]]), np.array([[0]]))
iter = dataset.make_initializable_iterator()
features, labels = iter.get_next()
with tf.Session() as sess:
#     initialise iterator with train data
    sess.run(iter.initializer, feed_dict={ x: train_data[0], y: train_data[1]})
    for i in range(EPOCHS):
        print (i, sess.run([features, labels]))
#     switch to test data
    sess.run(iter.initializer, feed_dict={ x: test_data[0], y: test_data[1]})
    print(sess.run([features, labels]))


print("\n\n")


a1 = np.arange(1, 25).reshape((4, 6))
print (a1)
idx = np.arange(2, 4)
a1_slice = a1[ :, idx]
print (a1_slice)
