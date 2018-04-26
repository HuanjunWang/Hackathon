


class RLAgent(object):
    def __init__(self, lr, state_len):
        self.state_h = tf.placeholder(tf.float32, shape=[None, state_len], name="States")
        self.action_h = tf.placeholder(tf.int32, shape=[None], name="Actions")
        self.reward_h = tf.placeholder(tf.float32, shape=[None], name="Rewards")

        l1 = slim.fully_connected(inputs=self.state_h, num_outputs=40, biases_initializer=None,
                                  activation_fn=tf.nn.relu)
        l2 = slim.fully_connected(inputs=l1, num_outputs=32, activation_fn=tf.nn.relu, biases_initializer=None)
        output = slim.fully_connected(inputs=l2, num_outputs=3, activation_fn=tf.nn.softmax, biases_initializer=None, )
        with tf.name_scope("RLOpt"):
            indexes = tf.range(0, tf.shape(output)[0]) * tf.shape(output)[1] + self.action_h + 1
            responsible_outputs = tf.gather(tf.reshape(output, [-1]), indexes)
            loss = -tf.reduce_mean(tf.log(responsible_outputs) * self.reward_h)
            # loss = tf.reduce_mean((1/responsible_outputs-1) * self.reward_h)
            # loss = - tf.reduce_mean(tf.sin(responsible_outputs) * self.reward_h)
            # loss = - tf.reduce_mean(responsible_outputs * self.reward_h)
            # loss = - tf.reduce_mean(tf.log(responsible_outputs) * (self.reward_h))
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        self.action_o = tf.argmax(output, axis=1) - 1
        self.update = optimizer.minimize(loss)


def run_with_rl(env):
    max_loop = 300000
    epsilon = 0.45
    epsilon_delta = 0.00002
    learning_rate = 0.01
    gamma = 0.99
    agent = RLAgent(lr=learning_rate, state_len=ENV.STATE_LEN)

    with tf.Session() as sess:
        tf.summary.FileWriter("./log", sess.graph)
        tf.global_variables_initializer().run()
        total_reward2 = 0
        for i in range(1, max_loop):
            state = env.reset()
            total_reward = 0
            game_over = False
            states = []
            actions = []
            rewards = []
            # while not game_over:
            while not game_over:
                states.append(state)
                if np.random.rand() > epsilon - i * epsilon_delta:
                    action = sess.run(agent.action_o, feed_dict={agent.state_h: [state]})[0]
                else:
                    action = np.random.randint(-1, 2)

                actions.append(action)
                next_state, reward, game_over = env.step(action)
                rewards.append(reward)
                state = next_state
                total_reward2 += reward

            for j in reversed(range(len(rewards) - 1)):
                rewards[j] += rewards[j + 1] * gamma

            sess.run(agent.update, feed_dict={agent.state_h: states, agent.action_h: actions, agent.reward_h: rewards})

            if i % 10 == 0:
                print("Round:", i, "  Total Reward:", total_reward2)
                total_reward2 = 0

            if i % 300 == 0:
                state = env.reset(0)
                total_reward = 0
                # while not game_over:
                for part in range(200000):
                    action = sess.run(agent.action_o, feed_dict={agent.state_h: [state]})[0]
                    state, reward, game_over = env.step(action)
                    total_reward += reward
                print("Test Round:", i, "  Total Reward:", total_reward)