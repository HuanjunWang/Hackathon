import pickle
import pprint

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
import math

from matplotlib import pyplot as plt


class Samples:
    DELAY_FILE = 'delay.txt'
    SMALL_FILE = 'small.txt'
    FEATURE_LEN = 100
    FEATURE_GAP = 5
    SMALL_NUM = 300000

    def __init__(self):
        self.raw = None
        self.X = []
        self.y = []
        self.Xval = []
        self.yval = []
        self.Xtest = []
        self.ytest = []
        self.load_data()

    def load_data(self):
        if not os.path.isfile(self.SMALL_FILE):
            with open(self.DELAY_FILE, "rb") as bf, open(self.SMALL_FILE, "wb") as sf:
                data = pickle.load(bf)
                pickle.dump(data[0:self.SMALL_NUM], sf)

        with open(self.SMALL_FILE, 'rb') as sf:
            self.raw = pickle.load(sf)

        for i in range(self.FEATURE_LEN + self.FEATURE_GAP, len(self.raw)):
            self.X.append(self.raw[i - self.FEATURE_LEN - self.FEATURE_GAP: i - self.FEATURE_GAP])
            self.y.append([self.raw[i]])

        slen = len(self.X)

        self.Xtest = self.X[math.floor(slen * 0.8): slen]
        self.ytest = self.y[math.floor(slen * 0.8): slen]

        self.Xval = self.X[math.floor(slen * 0.6): math.floor(slen * 0.8)]
        self.yval = self.y[math.floor(slen * 0.6): math.floor(slen * 0.8)]

        self.X = self.X[0:math.floor(slen * 0.6)]
        self.y = self.y[0:math.floor(slen * 0.6)]

    def print(self):
        print("Data Samples info:")
        print("X len :", len(self.X))
        print("y len :", len(self.y))
        print("Xvar len :", len(self.Xval))
        print("yvar len :", len(self.yval))
        print("Xtest len :", len(self.Xtest))
        print("ytest len :", len(self.ytest))
        # pprint.pprint(self.X[:100])
        # pprint.pprint(self.y[:100])


class NN:
    lr = 0.001
    info_loop = 10

    def __init__(self, samples, model=None, node_num=300):
        self.samples = samples
        self.model = model
        self.node_num = node_num

        self.state = tf.placeholder(tf.float32, (None, self.samples.FEATURE_LEN), name="State")
        self.y = tf.placeholder(tf.float32, (None, 1), name="y")
        with tf.name_scope("NN"):
            self.l1 = slim.fully_connected(self.state, self.node_num)  # , activation_fn=tf.sigmoid)
            self.l2 = slim.fully_connected(self.l1, self.node_num)  # , activation_fn=tf.sigmoid)
            self.l3 = slim.fully_connected(self.l2, self.node_num, activation_fn=tf.sigmoid)
            self.l4 = slim.fully_connected(self.l3, self.node_num)  # , activation_fn=tf.sigmoid)
            self.l5 = slim.fully_connected(self.l4, self.node_num)  # , activation_fn=tf.sigmoid)
            self.out_put = slim.fully_connected(self.l2, 1, activation_fn=None, biases_initializer=None)

        self.predict = tf.round(self.out_put)
        self.award = tf.reduce_sum(tf.cast(tf.abs(self.predict - self.y) <= 1, tf.float32))

        self.lose = tf.reduce_mean((self.out_put - self.y) ** 2)

        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.opt2 = tf.train.AdamOptimizer(learning_rate=self.lr)

        self.opt_op = self.opt2.minimize(self.lose)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.lose_his = []

    def verify(self):
        with tf.Session() as sess:
            if self.model is not None and os.path.isfile(self.model + ".meta"):
                print("Restore Saver")
                self.saver.restore(sess, self.model)

            lose = sess.run(self.lose, feed_dict={self.state: self.samples.X, self.y: self.samples.y})
            lose_val = sess.run(self.lose, feed_dict={self.state: self.samples.Xval, self.y: self.samples.yval})
            lose_test = sess.run(self.lose, feed_dict={self.state: self.samples.Xtest, self.y: self.samples.ytest})

            award = sess.run(self.award, feed_dict={self.state: self.samples.X, self.y: self.samples.y})
            award_val = sess.run(self.award, feed_dict={self.state: self.samples.Xval, self.y: self.samples.yval})
            award_test = sess.run(self.award, feed_dict={self.state: self.samples.Xtest, self.y: self.samples.ytest})

            print("lose:%f lose_val:%f, lose_test:%f" % (lose, lose_val, lose_test))
            # print("award:%f award_val:%f, award_test:%f" % (award, award_val, award_test))
            print("%f :%f :%f" % (
                award / len(self.samples.X), award_val / len(self.samples.Xval), award_test / len(self.samples.Xtest)))

    def run_training(self, lr=None):
        if lr is not None:
            print("Training using lr=", lr)
            self.lr = lr

        with tf.Session() as sess:
            sess.run(self.init)
            if self.model is not None and os.path.isfile(self.model + ".meta"):
                print("Restore Saver")
                self.saver.restore(sess, self.model)

            try:
                for i in range(50):
                    [opt, lose] = sess.run([self.opt_op, self.lose],
                                           feed_dict={self.state: self.samples.X, self.y: self.samples.y})
                    self.lose_his.append(lose)

                    if (i + 1) % self.info_loop == 0:
                        print("round:", i + 1, " lose:", lose)

            finally:
                print("Saver save")
                if self.model is not None:
                    self.saver.save(sess, self.model)
                with open(self.model + ".lose.history", "wb") as lose_file:
                    pickle.dump(self.lose_his, lose_file)

        return self.lose_his


if __name__ == "__main__":
    sample = Samples()
    sample.print()

    lose_history = []
    for nodes in [100]:
        for lr in [0.003]:
            model_name = "./saver/l5_%d_lr_%.6f_adam_model_2" % (nodes, lr)
            nn = NN(sample, model=model_name, node_num=nodes)

            lh = nn.run_training(lr=lr)
            lose_history.append(lh)
            plt.plot(range(len(lh)), lh, label="lr=%.4f node = %d" % (lr, nodes))
            nn.verify()

    leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.show()
