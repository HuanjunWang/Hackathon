import pickle
import numpy as np


class ENV(object):
    DELAY_FILE = 'delay.txt'
    STATE_LEN = 10
    MARGIN = 5
    WINDOW = 1

    def __init__(self):
        self.data = None
        self.index = None
        self.X = None
        self.Y = None
        self.TSize = None
        self.load_data()
        self.reset()

    def load_data(self):
        with open(self.DELAY_FILE, "rb") as f:
            self.data = pickle.load(f)
            self.X = []
            self.Y = []
            for i in range(ENV.STATE_LEN + 100000, len(self.data)):
                self.X.append(self.data[i - ENV.STATE_LEN - ENV.MARGIN:i-ENV.MARGIN])
                self.Y.append(self.data[i])
            self.TSize = len(self.X)

            print("Total PGSL messages:", len(self.data), "Training Samples:", self.TSize)

    def reset(self, index=None):
        if index is None:
            self.index = self.MARGIN + self.STATE_LEN
        else:
            if index < self.MARGIN + self.STATE_LEN or index > len(self.data):
                raise IndexError("Index out our delay data range")
            self.index = index
        return self.data[self.index - self.MARGIN - self.STATE_LEN:self.index - self.MARGIN]

    def step(self, advance):
        if np.abs(advance - self.data[self.index]) <= self.WINDOW:
            reward = 1
        else:
            reward = 0
        self.index += 1
        end = (self.index == len(self.data))

        return self.data[self.index - self.MARGIN - self.STATE_LEN:self.index - self.MARGIN], reward, end


if __name__ == "__main__":

    def run_with_fix_advance(env, adv, number=100000, start=0):
        env.reset()
        total_reward = 0

        for i in range(number):
            state, reward, end = env.step(advance=adv)
            total_reward += reward
            if end:
                break
        print("[Fix Adv]Messages:%d ADV:%d Total Reward:%d" % (number, adv, total_reward))


    env = ENV()
    run_with_fix_advance(env, adv=-2)
    run_with_fix_advance(env, adv=-1)
    run_with_fix_advance(env, adv=0)
    run_with_fix_advance(env, adv=1)
    run_with_fix_advance(env, adv=2)
    run_with_fix_advance(env, adv=3)
    run_with_fix_advance(env, adv=4)
    run_with_fix_advance(env, adv=5)
    run_with_fix_advance(env, adv=6)
    run_with_fix_advance(env, adv=7)
    run_with_fix_advance(env, adv=8)
