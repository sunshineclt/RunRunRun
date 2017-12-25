from ReplayBuffer import ReplayBuffer
import tensorflow as tf
import tensorflow.contrib.layers as layers
import time
import numpy as np
from utils.utils import TensorFlowFunction as tffunction
from Noise import OneFsqNoise

path_model = './model/'
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

        self.train_op, self.inference, self.update_target = self.build_train()
        self.summary_writer.add_graph(self.sess.graph)

        self.reward_tensorboard = tf.Variable(0, name='reward_tensorboard', dtype=tf.float32)
        self.reward_summary = tf.summary.scalar('Reward', reward_tensorboard)

        self.sess.run(tf.global_variables_initializer())
        self.update_target(1)

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
        # print(len(aw))
        # print(len(atw))
        # print(len(cw))
        # print(len(ctw))
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
        def train(memory):
            [s1d, a1d, r1d,isdoned, s2d] = memory
            result = self.sess.run([critic_loss, actor_loss, critic_train, actor_train, shift_critic, shift_actor],
                                   feed_dict={
                                       s1: s1d,
                                       a1: a1d,
                                       r1: r1d,
                                       isdone: isdoned,
                                       s2: s2d,
                                       tau: 1e-3
                                   })
            return result[0], result[1]

        inference = tffunction(self.sess, [s1], [a_infer, q_infer])
        update_target = tffunction(self.sess, [tau], [shift_critic, shift_actor])

        return train, inference, update_target

    def act(self, observation, current_noise=None):
        obs = np.reshape(observation, (1, len(observation)))
        actions, q = self.inference(obs)
        actions = actions[0]
        q = q[0]

        # if current_noise is not None:
        #     disp_actions = (actions - 0.5) / 0.5
        #     disp_actions = disp_actions * 5 + np.arange(self.action_dims) * 12.0 + 30
        #     noise = current_noise * 5 - np.arange(self.action_dims) * 12.0 - 30

        return actions, q

    def train_once(self, batch_size=64):
        if self.replay_buffer.size() > 2000:
            experiences = self.replay_buffer.sample_batch(batch_size)
            self.train_op(experiences)

    def play(self, env, noise_level, episode_index, max_steps=50000):
        timer = time.time()
        # noise_source = OneFsqNoise()
        # noise_source.skip = 4  # frequency adjustment
        # for j in range(200):
        #     noise_source.one((self.action_dims, ), noise_level)

        noise_phase = int(np.random.uniform() * 999999)

        steps = 0
        total_reward = 0
        total_q = 0

        s1 = env.reset()
        while steps <= max_steps:
            steps += 1

            phased_noise_annel_duration = 100
            phased_noise_amplitude = ((-noise_phase - steps) % phased_noise_annel_duration) / phased_noise_annel_duration
            exploration_noise = np.random.normal(size=(self.action_dims, )) * noise_level * phased_noise_amplitude

            a1, q_value = self.act(s1)
            total_q += q_value

            exploration_noise *= 0.5
            a1 += exploration_noise
            a1 = self.clamer(a1)
            a1_out = a1  # just in case env.step changes a1

            s2, r1, done, _info = env.step(a1_out)
            isdone = 1 if done else 0
            total_reward += r1
            if self.training:
                self.replay_buffer.add((s1, a1, r1, isdone, s2))
                self.train_once()
            if done:
                break
            s1 = s2

        totaltime = time.time() - timer
        print('episode {} done in {} steps in {:.2f} sec, {:.4f} sec/step, total reward :{:.2f}'.format(
            episode_index, steps, totaltime, totaltime / steps, total_reward
        ))
        self.sess.run(tf.assign(self.reward_tensorboard, total_reward))
        self.summary_writer.add_summary(self.sess.run(self.reward_summary), episode_index)

    def clamer(self, actions):
        return np.clip(actions, a_max=1, a_min=0)

    def save_checkpoints(self):
        saver = tf.train.Saver()
        saver.save(self.sess, path_model)

    def load_checkpoints(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, path_model)

