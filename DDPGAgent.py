from ReplayBuffer import ReplayBuffer
import tensorflow as tf
import tensorflow.contrib.layers as layers
import time
import numpy as np
from utils.utils import TensorFlowFunction as tffunction

path_model = 'model.ckpt'
tensorboard_path = 'tb_path'


class DDPGAgent:
    def __init__(self,
                 state_dims,
                 action_dims,
                 gamma=0.99):
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(1000000)

        self.sess = tf.Session()
        self.summary_writer = tf.summary.FileWriter(tensorboard_path)

        self.create_actor_network("actor_now")  # Just creating shared network
        self.create_actor_network("actor_target")  # Just creating shared network
        self.create_critic_network("critic_now")  # Just creating shared network
        self.create_critic_network("critic_target")  # Just creating shared network

        self.train_op, self.inference, self.sync_target = self.build_train()
        self.summary_writer.add_graph(self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.sync_target(1)

        self.training = True

    def create_actor_network(self, variable_scope, state_tensor=None, reuse=None):
        with tf.variable_scope(variable_scope, reuse=reuse):
            if state_tensor is None:
                state_tensor = tf.placeholder(shape=[None, self.state_dims], dtype=tf.float32)
            out = layers.fully_connected(state_tensor,
                                         num_outputs=128,
                                         activation_fn=tf.nn.selu)
            out = layers.fully_connected(out,
                                         num_outputs=128,
                                         activation_fn=tf.nn.selu)
            out = layers.fully_connected(out,
                                         num_outputs=64,
                                         activation_fn=tf.nn.selu)
            out = layers.fully_connected(out,
                                         num_outputs=64,
                                         activation_fn=tf.nn.selu)
            out = layers.fully_connected(out,
                                         num_outputs=self.action_dims,
                                         activation_fn=tf.nn.tanh)
            out = out * 0.5 + 0.5  # tanh result in [-1, 1] while we need [0, 1]
            return out

    def create_critic_network(self, variable_scope, state_tensor=None, action_tensor=None, reuse=None):
        with tf.variable_scope(variable_scope, reuse=reuse):
            if state_tensor is None:
                state_tensor = tf.placeholder(shape=[None, self.state_dims], dtype=tf.float32)
            if action_tensor is None:
                action_tensor = tf.placeholder(shape=[None, self.action_dims], dtype=tf.float32)
            out = layers.fully_connected(state_tensor,
                                         num_outputs=128,
                                         activation_fn=tf.nn.selu)
            out = tf.concat([out, action_tensor], 1)
            out = layers.fully_connected(out,
                                         num_outputs=128,
                                         activation_fn=tf.nn.selu)
            out = layers.fully_connected(out,
                                         num_outputs=64,
                                         activation_fn=tf.nn.selu)
            out = layers.fully_connected(out,
                                         num_outputs=64,
                                         activation_fn=tf.nn.selu)
            out = layers.fully_connected(out,
                                         num_outputs=1,
                                         activation_fn=None)
            return out

    def build_train(self):
        s1 = tf.placeholder(tf.float32, shape=[None, self.state_dims])
        a1 = tf.placeholder(tf.float32, shape=[None, self.action_dims])
        r1 = tf.placeholder(tf.float32, shape=[None, 1])
        isdone = tf.placeholder(tf.float32, shape=[None, 1])
        s2 = tf.placeholder(tf.float32, shape=[None, self.state_dims])

        # critic loss
        a2 = self.create_actor_network("actor_target", state_tensor=s2, reuse=True)
        q2 = self.create_critic_network("critic_target", state_tensor=s2, action_tensor=a2, reuse=True)
        q1_target = r1 + (1 - isdone) * self.gamma * q2
        q1_predicted = self.create_critic_network("critic_now", state_tensor=s1, action_tensor=a1, reuse=True)
        critic_loss = tf.reduce_mean((q1_target - q1_predicted) ** 2)

        # actor loss
        a1_predicted = self.create_actor_network("actor_now", state_tensor=s1, reuse=True)
        q1_a1_predicted = self.create_critic_network("critic_now", state_tensor=s1, action_tensor=a1_predicted, reuse=True)
        actor_loss = tf.reduce_mean(-q1_a1_predicted)

        # target network update
        tau = tf.placeholder(dtype=tf.float32, shape=())
        aw = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor_now")
        atw = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor_target")
        cw = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic_now")
        ctw = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic_target")
        print(len(aw))
        print(len(atw))
        print(len(cw))
        print(len(ctw))
        shift_actor = [tf.assign(atw[i], aw[i] * tau + atw[i] * (1 - tau)) for i, _ in enumerate(aw)]
        shift_critic = [tf.assign(ctw[i], cw[i] * tau + ctw[i] * (1 - tau)) for i, _ in enumerate(cw)]

        # infer
        a_infer = a1_predicted
        q_infer = q1_a1_predicted

        # optimize
        opt = tf.train.AdamOptimizer(3e-5)
        opt_actor, opt_critic = opt, opt
        critic_train = opt_critic.minimize(critic_loss, var_list=cw)
        actor_train = opt_actor.minimize(actor_loss, var_list=aw)

        # define tffunctions
        train = tffunction(self.sess,
                           [s1, a1, r1, isdone, s2],
                           [critic_loss, actor_loss, critic_train, actor_train, shift_critic, shift_actor])
        inference = tffunction(self.sess, [s1], [a_infer, q_infer])
        sync_target = tffunction(self.sess, [tau], [shift_critic, shift_actor])

        return train, inference, sync_target

    def act(self, observation):
        obs = np.reshape(observation, (1, len(observation)))
        actions, q = self.inference(obs)
        actions = actions[0]
        q = q[0]
        return actions, q

    def train_once(self):
        memory = self.replay_buffer
        batch_size = 64

        if memory.size() > 2000:
            [s1, a1, r1, isdone, s2] = memory.sample_batch(batch_size)
            self.train_op(s1, a1, r1, isdone, s2)

    def play(self, env, max_steps=50000):
        timer = time.time()
        steps = 0
        total_reward = 0
        total_q = 0

        s1 = env.reset()
        while steps <= max_steps:
            steps += 1
            a1, q_value = self.act(s1)
            total_q += q_value
            a1 = self.clamer(a1)
            s2, r1, done, _info = env.step(a1)
            isdone = 1 if done else 0
            total_reward += r1
            if self.training:
                self.replay_buffer.add((s1, a1, r1, isdone, s2))
                self.train_once()
            if done:
                break

        totaltime = time.time() - timer
        print('episode done in {} steps in {:.2f} sec, {:.4f} sec/step, total reward :{:.2f}'.format(
            steps, totaltime, totaltime / steps, total_reward
        ))

    def clamer(self, actions):
        return np.clip(actions, a_max=1, a_min=0)

    def save_checkpoints(self):
        saver = tf.train.Saver()
        saver.save(tf.get_default_session(), path_model)

    def load_checkpoints(self):
        saver = tf.train.Saver()
        saver.restore(tf.get_default_session(), path_model)

