import threading
import time

from osim.env import RunEnv

from DDPGAgent import DDPGAgent
from farmer import Farmer
from observation_processor import processed_dims
from utils.EnvWarpper import FastEnv
from utils.triggerbox import TriggerBox

if __name__ == "__main__":
    tempenv = RunEnv(visualize=False)
    agent = DDPGAgent(processed_dims, tempenv.action_space.shape[0], gamma=0.99)

    farmer = Farmer()

    noise_level = 0.5
    noise_decay_rate = 0.001
    noise_floor = 0
    noiseless = 0.0001
    global_episode_index = 0

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


    def play_one_episode(env, nl, episode_index):
        fast_env = FastEnv(env, 3)  # 4 is skip factor
        agent.play(fast_env, noise_level=nl, episode_index=episode_index)
        env.rel()
        del fast_env

    def play_async(env, nl, episode_index):
        thread = threading.Thread(target=play_one_episode, args=(env, nl, episode_index))
        thread.daemon = True
        thread.start()

    def play_if_available(nl, episode_index):
        while True:
            remote_env = farmer.acq_env()
            if not remote_env:  # There is no free environment
                pass
            else:
                play_async(remote_env, nl, episode_index)
                break

    def play_repeat(episode_number):
        global stop_flag, noise_level, global_episode_index
        for i in range(episode_number):
            if stop_flag:
                stop_flag = False
                print("(play_repeat) stop signal received, stop at episode", i + 1)
                break

            noise_level *= (1 - noise_decay_rate)
            noise_level = max(noise_floor, noise_level)
            nl = noise_level if i % 4 == 0 else noiseless
            print('episode {}/{}, noise_level: {}'.format(
                i+1, episode_number, nl
            ))

            play_if_available(nl, global_episode_index)
            global_episode_index += 1

            time.sleep(0.05)
            if i % 100 == 0 or i == episode_number - 1:
                save()

    def test(skip=4):
        test_env = RunEnv(visualize=True, max_obstacles=0)
        fast_env = FastEnv(test_env, skip)  # 4 is skip factor
        agent.training = False
        agent.play(fast_env, noise_level=1e-11, episode_index=-1)
        agent.training = True
        del test_env

    def save():
        agent.save_checkpoints()
        pass

    def load():
        agent.load_checkpoints()
        pass
