#!/usr/bin/env python

"""
Example usage:
    python run_bc.py ./hw1/saved_models/Ant-v2-100000.ckpt Ant-v2 --render --num_rollouts 20

"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

from nn import *


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('saved_model', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    # Dims for Ant-v2 (111,8)
    input_ph, _, output_pred = create_model(111, 8)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # restore saved model
        saver.restore(sess, args.saved_model)

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                obs = obs.reshape((1, -1))
                action = sess.run(output_pred, feed_dict={input_ph: obs})
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        # expert_data = {'observations': np.array(observations),
        #                'actions': np.array(actions)}



if __name__ == '__main__':
    main()
