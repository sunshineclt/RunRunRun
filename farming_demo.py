# this file is created to demonstrate how one can construct a client program that obtains RunEnv instances from a farm.
# Adapted from https://github.com/ctmakro/stanford-osrl


# instructions to start a farm is located in README.md

from farmer import Farmer
import numpy as np
import time

# singleton
farmer = Farmer()


# reset all your farms (usually not needed.)
def refarm():
    global farmer
    del farmer
    farmer = Farmer()


# step thru the environment step by step.
def playonce(env):
    ob = env.reset()
    step = 0
    for i in range(500):
        o, r, d, _ = env.step([j for j in np.random.uniform(size=(18,))])
        step += 1
        print('envid', env.custom_id, 'step', step)
        if d:
            break

    # release the env
    env.rel()


def play_ignore(env):
    import threading as th
    t = th.Thread(target=playonce, args=(env,))
    t.daemon = True
    t.start()
    # ignore and return, let the thread run for itself.


# acquire the env from farm.
def play_if_available(cid):
    while True:
        remote_env = farmer.acq_env()
        # just replace occurences of env = RunEnv() with the line above.

        # the remote_env obtained here is preconfigured in `farm.py`. please modify `farm.py` to change the
        # parameters like difficulty or num_obstacles.

        if not remote_env:  # no free environment
            # time.sleep(0.1)
            pass
        else:
            remote_env.custom_id = cid
            play_ignore(remote_env)
            break


if __name__ == '__main__':
    refarm()

    for i in range(5):
        play_if_available(i)

    time.sleep(20)  # at this point, 5 instances should be running in parallel.
