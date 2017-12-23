from ReplayBuffer import ReplayBuffer
import tensorflow as tf
import tensorflow.contrib.layers as layers
from utils.utils import TensorFlowFunction as tffunction


class DDPGAgent:
    def __init__(self,
                 state_dims,
                 action_dims,
                 gamma=0.99):
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(1000000)

        self.actor = self.create_actor_network("actor")
        self.actor_target = self.create_actor_network("actor_target")
        self.critic = self.create_critic_network("critic")
        self.critic_target = self.create_critic_network("critic_target")

    def create_actor_network(self, variable_scope):
        with tf.variable_scope(variable_scope):
            state_tensor = tf.placeholder(shape=[None, self.state_dims], dtype=tf.float32)
            out = layers.fully_connected(self.state_tensor,
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
                                         num_outputs=64,
                                         activation_fn=tf.nn.selu)
            out = layers.fully_connected(out,
                                         num_outputs=self.action_dims,
                                         activation_fn=tf.nn.tanh)
            return state_tensor, out

    def create_critic_network(self, variable_scope):
        with tf.variable_scope(variable_scope):
            state_tensor = tf.placeholder(shape=[None, self.state_dims], dtype=tf.float32)
            action_tensor = tf.placeholder(shape=[None, self.action_dims], dtype=tf.float32)
            out = layers.fully_connected(state_tensor,
                                         num_outputs=128,
                                         activation_fn=tf.nn.selu)
            out = tf.concat([out, action_tensor], axis=1)
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
            return state_tensor, action_tensor, out

    def build_train(self):

