from ENV import ENV
import numpy as np

def run_with_average_value(env, number=100000, start=0):
    state = env.reset()
    total_reward = 0
    for i in range(number):
        adv = int(np.around(np.mean(state[-1])))
        state, reward, end = env.step(advance=adv)
        total_reward += reward
        if end:
            break

    print("[Average] Messages:%d number:%d Total Reward:%d" % (number, env.STATE_LEN, total_reward))


if __name__ == "__main__":
    env = ENV()
    run_with_average_value(env)


