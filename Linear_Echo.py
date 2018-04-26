import tensorflow as tf
from ENV import ENV


def run_with_linear(env, number=100000, lr=0.01):
    training_epochs = 2000
    training_step = 1000
    display_step = 100
    verify_step = 500


    state_h = tf.placeholder(tf.float32, shape=[None, ENV.STATE_LEN], name="States")
    delay = tf.placeholder(tf.float32, shape=[None], name="delay")
    weight = tf.get_variable("weight", [ENV.STATE_LEN])
    bias = tf.get_variable("bias", [1])
    pred_delay = tf.add(tf.reduce_sum(tf.multiply(state_h, weight), 1), bias)
    pows = tf.pow(pred_delay - delay, 2)
    cost = tf.reduce_sum(pows) / (2 * number)

    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        tf.summary.FileWriter("./log", sess.graph)
        sess.run(init)
        for epoch in range(training_epochs):

            i = 0
            while i < env.TSize:
                sess.run(optimizer, feed_dict={state_h: env.X[i:i+training_step], delay: env.Y[i:i+training_step]})
                i += training_step

            if (epoch + 1) % display_step == 0:
                c = sess.run(cost, feed_dict={state_h: env.X[:number], delay: env.Y[:number]})
                print("Epoch:", '%06d' % (epoch + 1), "cost=", "{:.9f}".format(c), \
                      "W=", sess.run(weight), "b=", sess.run(bias))

            if (epoch + 1) % verify_step == 0:
                state = env.reset()
                total_reward = 0
                for i in range(number):
                    adv = sess.run(pred_delay, feed_dict={state_h: [state]})
                    state, reward, end = env.step(advance=adv)
                    total_reward += reward
                    if end:
                        break
                print("[SL:Linear] Messages:%d number:%d Total Reward:%d" % (number, env.STATE_LEN, total_reward))


if __name__ == "__main__":
    env = ENV()
    run_with_linear(env)
