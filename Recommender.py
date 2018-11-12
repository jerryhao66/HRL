from __future__ import absolute_import
from __future__ import division

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import math

class RecommenderNetwork(object):

    def __init__(self, sess, num_items, args):
        self.num_items = num_items
        self.global_step = tf.Variable(0, trainable=False, name="Attention_step")
        self.learning_rate = args.recommender_lr
        self.embedding_size = args.recommender_embedding_size
        self.weight_size = args.recommender_weight_size
        self.alpha = args.alpha
        self.beta = args.beta
        self.algorithm = args.algorithm
        self.regs = args.regs
        self.tau = args.recommender_tau
        self.lambda_bilinear = args.regs[0]
        self.gamma_bilinear = args.regs[1]
        self.eta_bilinear = args.regs[2]
        self.sess = sess

        self.num_other_variables = len(tf.trainable_variables())
        self.user_input, self.num_idx, self.item_input, self.output,_,_,_ = self.create_recommender_network("Active")
        self.network_params = tf.trainable_variables()[self.num_other_variables:]

        self.target_user_input, self.target_num_idx, self.target_item_input, self.target_output, self.target_Q_, self.target_Q, self.target_W = self.create_recommender_network("Target")
        self.target_network_params = tf.trainable_variables()[len(self.network_params) + self.num_other_variables:]

        # delayed updating recommender network ops
        self.update_target_network_params = \
            [self.target_network_params[i].assign( \
                tf.multiply(self.network_params[i], self.tau) + \
                tf.multiply(self.target_network_params[i], 1 - self.tau)) \
                for i in range(len(self.target_network_params))]

        # network parameters --> target network parameters
        self.assign_target_network_params = \
            [self.target_network_params[i].assign( \
                self.network_params[i]) for i in range(len(self.target_network_params))]

        # target network parameters -->  network parameters
        self.assign_active_network_params = \
            [self.network_params[i].assign( \
                self.target_network_params[i]) for i in range(len(self.network_params))]

        self.one_minus_output = 1.0 - self.target_output
        self.reward_output_concat = tf.concat([self.one_minus_output, self.target_output], 1)
        self.classes = tf.constant(2)
        self.labels = tf.placeholder(tf.int32, shape=[None, 1])  # the ground truth

        self.labels_reduce_dim = tf.reduce_sum(self.labels, 1)
        self.one_hot = tf.one_hot(self.labels_reduce_dim, self.classes)

        self.reward = tf.log(tf.reduce_sum((self.reward_output_concat * self.one_hot + 1e-15), 1))
        self.l2loss = 0
        self.l2loss = self.lambda_bilinear * tf.reduce_sum(tf.square(self.target_Q)) + \
                      self.gamma_bilinear * tf.reduce_sum(tf.square(self.target_Q_)) + \
                      self.eta_bilinear * tf.reduce_sum(tf.square(self.target_W))

        self.loss = tf.losses.log_loss(self.labels, self.target_output)
        self.loss += self.l2loss
        self.gradients = tf.gradients(self.loss, self.target_network_params)
        # self.gradients = tf.clip_by_value(gradients, 1e-5, 1 - 1e-5)

        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate, initial_accumulator_value=1e-8).apply_gradients(
            zip(self.gradients, self.network_params), global_step=self.global_step)

        # total variables
        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)
        self.num_network_params = len(self.network_params)
        self.num_target_network_params = len(self.target_network_params)


    def _create_variables(self, scope):
        with tf.name_scope(scope):  # The embedding initialization is unknown now
            self.c1 = tf.Variable(
                tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01), name='c1',
                dtype=tf.float32, trainable=True)
            self.c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2')
            self.embedding_Q_ = tf.concat([self.c1, self.c2], 0, name='embedding_Q_')
            self.embedding_Q = tf.Variable(
                tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                name='embedding_Q', dtype=tf.float32, trainable=True)
            self.bias = tf.Variable(tf.zeros(self.num_items), name='bias', trainable=True)

            # Variables for attention
            if self.algorithm == 0:
                self.W = tf.Variable(tf.truncated_normal(shape=[self.embedding_size, self.weight_size], mean=0.0,
                                                         stddev=tf.sqrt(
                                                             tf.div(2.0, self.weight_size + self.embedding_size))),
                                     name='Weights_for_MLP', dtype=tf.float32, trainable=True)
            else:
                self.W = tf.Variable(tf.truncated_normal(shape=[2 * self.embedding_size, self.weight_size], mean=0.0,
                                                         stddev=tf.sqrt(tf.div(2.0, self.weight_size + (
                                                             2 * self.embedding_size)))), name='Weights_for_MLP',
                                     dtype=tf.float32, trainable=True)
            self.b = tf.Variable(tf.truncated_normal(shape=[1, self.weight_size], mean=0.0, stddev=tf.sqrt(
                tf.div(2.0, self.weight_size + self.embedding_size))), name='Bias_for_MLP', dtype=tf.float32,
                                 trainable=True)
            self.h = tf.Variable(tf.ones([self.weight_size, 1]), name='H_for_MLP', dtype=tf.float32)

    def _attention_MLP(self, q_, scope):
        with tf.name_scope(scope):
            b = tf.shape(q_)[0]
            n = tf.shape(q_)[1]
            r = (self.algorithm + 1) * self.embedding_size

            MLP_output = tf.matmul(tf.reshape(q_, [-1, r]), self.W) + self.b  # (b*n, e or 2*e) * (e or 2*e, w) + (1, w)
            MLP_output = tf.nn.relu(MLP_output)

            A_ = tf.reshape(tf.matmul(MLP_output, self.h), [b, n])  # (b*n, w) * (w, 1) => (None, 1) => (b, n)

            # softmax for not mask features
            exp_A_ = tf.exp(A_)
            num_idx = tf.reduce_sum(self.num_idx, 1)
            mask_mat = tf.sequence_mask(num_idx, maxlen=n, dtype=tf.float32)  # (b, n)
            exp_A_ = mask_mat * exp_A_
            exp_sum = tf.reduce_sum(exp_A_, 1, keep_dims=True)  # (b, 1)
            exp_sum = tf.pow(exp_sum, tf.constant(self.beta, tf.float32, [1]))


            self.A = tf.expand_dims(tf.div(exp_A_, exp_sum), 2)  # (b, n, 1)

            return tf.reduce_sum(self.A * self.embedding_q_, 1)

    def _create_inference(self, scope):
        with tf.name_scope(scope):
            self.user_input = tf.placeholder(tf.int32, shape=[None, None])  # the index of users
            self.num_idx = tf.placeholder(tf.float32, shape=[None, 1])  # the number of items rated by users
            self.item_input = tf.placeholder(tf.int32, shape=[None, 1])  # the index of items

            self.embedding_q_ = tf.nn.embedding_lookup(self.embedding_Q_, self.user_input)  # (b, n, e)
            self.embedding_q = tf.nn.embedding_lookup(self.embedding_Q, self.item_input)  # (b, 1, e)

            if self.algorithm == 0:
                self.embedding_p = self._attention_MLP(self.embedding_q_ * self.embedding_q, scope)
            else:
                n = tf.shape(self.user_input)[1]
                self.embedding_p = self._attention_MLP(
                    tf.concat([self.embedding_q_, tf.tile(self.embedding_q, tf.stack([1, n, 1]))], 2), scope)

            self.embedding_q = tf.reduce_sum(self.embedding_q, 1)
            self.bias_i = tf.nn.embedding_lookup(self.bias, self.item_input)
            self.coeff = tf.pow(self.num_idx, tf.constant(self.alpha, tf.float32, [1]))
            self.output = tf.sigmoid(
                self.coeff * tf.expand_dims(tf.reduce_sum(self.embedding_p * self.embedding_q, 1), 1) + self.bias_i)

    def create_recommender_network(self, scope):
        self._create_variables(scope)
        self._create_inference(scope)
        return self.user_input, self.num_idx, self.item_input, self.output, self.embedding_q_, self.embedding_q, self.W

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def assign_target_network(self):
        self.sess.run(self.assign_target_network_params)

    def assign_active_network(self):
        self.sess.run(self.assign_active_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars, self.num_network_params, self.num_target_network_params

    def getloss(self, user_input, num_idx, item_input, labels):
        feed_dict = {self.target_user_input: user_input, self.target_num_idx: num_idx,
                     self.target_item_input: item_input,
                     self.labels: labels}
        return self.sess.run(self.loss, feed_dict)

    def train(self, user_input, num_idx, item_input, labels):
        feed_dict = {self.target_user_input: user_input, self.target_num_idx: num_idx,
                     self.target_item_input: item_input,
                     self.labels: labels}
        return self.sess.run([self.loss, self.optimizer], feed_dict)

    def predict(self, test_u, test_num, test_i, test_l):
        feed_dict = {self.user_input: test_u, self.num_idx: test_num, self.item_input: test_i,
                     self.labels: test_l}

        return self.sess.run([self.target_output, self.loss], feed_dict)

    def predict_with_atteionts(self, test_u, test_num, test_i, test_l):
        feed_dict = {self.user_input: test_u, self.num_idx: test_num, self.item_input: test_i,
                     self.labels: test_l}

        return self.sess.run([self.target_output, self.A, self.loss], feed_dict)


    def get_course_embedding(self):
        course_embedding_user = self.sess.run(self.embedding_Q_)
        course_embedding_item = self.sess.run(self.embedding_Q)
        return np.array(course_embedding_user), np.array(course_embedding_item)


    def get_reward(self, user_input, num_idx, item_input, labels):
        feed_dict = {self.user_input: user_input, self.num_idx: num_idx, self.item_input: item_input, self.labels: labels}
        return self.sess.run(self.reward, feed_dict)

    def get_rewards(self, dataset):
        user_input, num_idx, item_input, labels, num_batch = dataset[0],dataset[1],dataset[2],dataset[3],dataset[4]
        batch_reward_likelihood = []
        for batch_index in range(num_batch):
            batched_user_input = np.array([u for u in user_input[batch_index]])
            batched_item_input = np.reshape(item_input[batch_index], (-1, 1))
            batched_label_input = np.reshape(labels[batch_index], (-1, 1))
            batched_num_idx = np.reshape(num_idx[batch_index], (-1,1))

            batch_reward = self.get_reward(batched_user_input, batched_num_idx, batched_item_input, batched_label_input)
            batch_reward_likelihood.append(batch_reward)

        return np.array(batch_reward_likelihood)












