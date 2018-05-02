import tensorflow as tf
from ENV import ENV
import tensorflow.contrib.slim as slim


def run_with_linear(env, number=100000, lr=0.0001):
    training_epochs = 200000
    training_step = 1000
    display_step = 50
    verify_step = 1000

    print("NN, lr = ",lr)
    state_h = tf.placeholder(tf.float32, shape=[None, ENV.STATE_LEN], name="States")
    delay = tf.placeholder(tf.float32, shape=[None], name="delay")

    l1 = slim.fully_connected(inputs=state_h, num_outputs=400, biases_initializer=None,
                              activation_fn=tf.nn.relu)
    l2 = slim.fully_connected(inputs=l1, num_outputs=1000,biases_initializer=None,
                              activation_fn=tf.nn.relu)
    l3 = slim.fully_connected(inputs=l2, num_outputs=100,biases_initializer=None,
                              activation_fn=tf.nn.relu)

    l4 = slim.fully_connected(inputs=l3, num_outputs=1, biases_initializer=None,
                              activation_fn=None)

    pred_delay = tf.reshape(l4, [-1])

    pows = tf.pow(pred_delay - delay, 2)
    cost = tf.reduce_sum(pows) / (2 * number)

    optimizer = tf.train.AdamOptimizer(lr).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        tf.summary.FileWriter("./log", sess.graph)
        sess.run(init)
        for epoch in range(training_epochs):
            i = 0
            while i < env.TSize:
                sess.run(optimizer, feed_dict={state_h: env.X[i:i + training_step], delay: env.Y[i:i + training_step]})
                i += training_step

            if (epoch + 1) % display_step == 0:
                c = sess.run(cost, feed_dict={state_h: env.VX, delay: env.VY})
                print("Epoch:", '%06d' % (epoch + 1), "cost=", "{:.9f}".format(c))

            if (epoch + 1) % verify_step == 0:
                state = env.reset()
                total_reward = 0
                for i in range(number):
                    adv = sess.run(pred_delay, feed_dict={state_h: [state]})
                    state, reward, end = env.step(advance=adv)
                    total_reward += reward
                    if end:
                        break
                print("[SL:NN] Messages:%d number:%d Total Reward:%d" % (number, env.STATE_LEN, total_reward))


if __name__ == "__main__":
    env = ENV()
    run_with_linear(env)
