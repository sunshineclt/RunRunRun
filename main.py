from osim.env import RunEnv
from observation_processor import processed_dims
from farmer import Farmer
from utils.triggerbox import TriggerBox
from utils.EnvWarpper import FastEnv
import threading
import time

if __name__ == "__main__":
    tempenv = RunEnv(visualize=False)
    agent = None
    farmer = Farmer()


    def refarm():
        global farmer
        del farmer
        farmer = Farmer()


    stop_flag = False


    def stopsim():
        global stop_flag
        print("Stop Simulation has been called! ")
        stop_flag = True


    tb = TriggerBox('Press a button to do something.',
                    ['Stop Simulation'],
                    [stopsim])


    def play_one_episode(env):
        fast_env = FastEnv(env, 3)  # skipcount = 3
        # agent.play(fast_env)
        env.rel()
        del fast_env


    def play_async(env):
        thread = threading.Thread(target=play_one_episode, args=(env))
        thread.daemon = True
        thread.start()


    def play_if_available():
        while True:
            remote_env = farmer.acq_env()
            if not remote_env:  # There is no free environment
                pass
            else:
                play_async(remote_env)
                break


    def play_repeat(episode):
        global stop_flag
        for i in range(episode):
            if stop_flag:
                stop_flag = False
                print("(play_repeat) stop signal received, stop at episode", i + 1)
                break

            print('episode', i + 1, '/', episode)
            play_if_available()

            time.sleep(0.05)
            if (i + 1) % 2000 == 0:
                save()


    def test():
        test_env = RunEnv(visualize=True, max_obstacles=0)
        # agent.play(test_env)


    def save():
        # agent.save_weights()
        pass

    def load():
        # agent.load_weights()
        pass
