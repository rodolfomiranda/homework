import pickle
import tensorflow as tf
import numpy as np

from nn import *

datafile_path = "./hw1/expert_data/Ant-v2.pkl"
env_name = "Ant-v2"
batch_size = 100
total_iter = 100000

# load training data
f = open(datafile_path, 'rb')
expert_data = pickle.load(f)
f.close()
expert_data['actions'] = expert_data['actions'].reshape((expert_data['actions'].shape[0], -1))

input_dim = expert_data['observations'].shape[1]
output_dim = expert_data['actions'].shape[1]

sess = tf.Session()

input_ph, output_ph, output_pred = create_model(input_dim, output_dim)

# define loss
loss = tf.losses.mean_squared_error(output_ph, output_pred)
opt = tf.train.AdamOptimizer().minimize(loss)

# initialize variables and prepare for training
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

# run training
for step in range(total_iter+1):
    indices = np.random.randint(low=0, high=len(expert_data['actions']), size=batch_size)
    input_batch = expert_data['observations'][indices]
    output_batch = expert_data['actions'][indices]
    _, currloss = sess.run(
        [opt, loss],
        feed_dict={
            input_ph: input_batch,
            output_ph: output_batch
        }
    )
    if step > 0 and step % 1000 == 0:
        print("{0:-8d} iter, loss: {1:.5f}".format(step, currloss))
        if step % 5000 == 0:
            saver.save(sess, "hw1/saved_models/" + env_name + "-" + str(step) + ".ckpt")