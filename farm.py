# farm.py
# a single instance of a farm.
# Adapted from https://github.com/ctmakro/stanford-osrl

# a farm should consist of a pool of instances
# and expose those instances as one giant callable class

import multiprocessing
import random
import threading
import time
import traceback
from multiprocessing import Process, Queue

ncpu = multiprocessing.cpu_count()


# # bind our custom version of pelvis too low judgement function to original env
# def bind_alternative_pelvis_judgement(runenv):
#     def is_pelvis_too_low(self):
#         return self.current_state[self.STATE_PELVIS_Y] < (0.5 if True else 0.65)
#
#     import types
#     runenv.is_pelvis_too_low = types.MethodType(is_pelvis_too_low, runenv)


# # use custom episode length.
# def use_alternative_episode_length(runenv):
#     runenv.spec.timestep_limit = 2000


# def runenv_with_alternative_obstacle_generation_scheme():
#     from osim.env import RunEnv
#     try:
#         from rosetta import tacos, psoas  # check if rosetta.py exists.
#     except:
#         return RunEnv
#     finally:
#         pass
#
#     class RunEnv2(RunEnv):
#         def generate_env(self, difficulty, seed, max_obstacles):
#             import numpy as np
#             # if seed is not None:
#             #     np.random.seed(seed) # seed the RNG if seed is provided
#
#             # obstacles
#             num_obstacles = 0
#             xs = []
#             ys = []
#             rs = []
#
#             num_episodes = len(psoas)
#             num_obstacles = int(len(tacos) / num_episodes)
#             record_index = np.random.choice(num_episodes)
#             print('num_obstacles', num_obstacles, 'num_episodes', num_episodes)
#
#             if 0 < difficulty:
#                 # num_obstacles = min(3, max_obstacles)
#                 # xs = np.random.uniform(1.0, 5.0, num_obstacles)
#                 # ys = np.random.uniform(-0.25, 0.25, num_obstacles)
#                 # rs = [0.05 + r for r in np.random.exponential(0.05, num_obstacles)]
#                 for n in range(num_obstacles):
#                     x = tacos[record_index * num_obstacles + n][0]
#                     y = tacos[record_index * num_obstacles + n][1]
#                     r = tacos[record_index * num_obstacles + n][2]
#                     xs.append(x)
#                     ys.append(y)
#                     rs.append(r)
#
#             # if 0 < difficulty and 3 < max_obstacles:
#             #     extra_obstacles = max(min(20, max_obstacles) - num_obstacles, 0)
#             #     xs = np.concatenate([xs,(np.cumsum(np.random.uniform(2.0, 4.0, extra_obstacles)) + 5)])
#             #     ys = np.concatenate([ys,np.random.uniform(-0.05, 0.25, extra_obstacles)])
#             #     rs = rs + [0.05 + r for r in np.random.exponential(0.05, extra_obstacles)]
#             #     num_obstacles = len(xs)
#
#             # ys = map(lambda xy: xy[0]*xy[1], list(zip(ys, rs)))
#
#             # muscle strength
#             rpsoas = 1
#             lpsoas = 1
#             if difficulty >= 2:
#                 # rpsoas = 1 - np.random.normal(0, 0.1)
#                 # lpsoas = 1 - np.random.normal(0, 0.1)
#                 # rpsoas = max(0.5, rpsoas)
#                 # lpsoas = max(0.5, lpsoas)
#                 lpsoas = psoas[record_index][0]
#                 rpsoas = psoas[record_index][1]
#
#             muscles = [1] * 18
#
#             # modify only psoas
#             muscles[self.MUSCLES_PSOAS_R] = rpsoas
#             muscles[self.MUSCLES_PSOAS_L] = lpsoas
#
#             obstacles = list(zip(xs, ys, rs))
#             obstacles.sort()
#
#             return {
#                 'muscles': muscles,
#                 'obstacles': obstacles
#             }
#
#     return RunEnv2


# separate process that holds a separate RunEnv instance.
# This has to be done since RunEnv() in the same process result in interleaved running of simulations.
def standalone_headless_isolated(pq, cq, plock):
    # locking to prevent mixed-up printing.
    plock.acquire()
    print('starting headless...', pq, cq)
    try:
        from osim.env import RunEnv
        # RunEnv = runenv_with_alternative_obstacle_generation_scheme()
        e = RunEnv(visualize=False, max_obstacles=0)
        # bind_alternative_pelvis_judgement(e)
        # use_alternative_episode_length(e)
    except Exception as err:
        print('error on start of standalone')
        traceback.print_exc()
        plock.release()
        return
    else:
        plock.release()

    def report(e):
        # a way to report errors ( since you can't just throw them over a pipe )
        # e should be a string
        print('(standalone) got error!!!')
        cq.put(('error', e))

    def floatify(np):
        return [float(np[i]) for i in range(len(np))]

    try:
        while True:
            msg = pq.get()
            # messages should be tuples,
            # msg[0] should be string

            # isinstance is dangerous, commented out
            # if not isinstance(msg,tuple):
            #     raise Exception('pipe message received by headless is not a tuple')

            if msg[0] == 'reset':
                o = e.reset(difficulty=0)
                cq.put(floatify(o))
            elif msg[0] == 'step':
                o, r, d, i = e.step(msg[1])
                o = floatify(o)  # floatify the observation
                cq.put((o, r, d, i))
            else:
                cq.close()
                pq.close()
                del e
                break
    except Exception as e:
        traceback.print_exc()
        report(str(e))

    return  # end process


# global process lock
plock = multiprocessing.Lock()
# global thread lock
tlock = threading.Lock()

# global id issurance
eid = int(random.random() * 100000)


def get_eid():
    global eid, tlock
    tlock.acquire()
    i = eid
    eid += 1
    tlock.release()
    return i


# class that manages the interprocess communication and expose itself as a RunEnv.
# reinforced: this class should be long-running. it should reload the process on errors.

class EnvironmentInstance:  # Environment Instance
    def __init__(self):
        self.occupied = False  # is this instance occupied by a remote client
        self.id = get_eid()  # what is the id of this environment
        self.pretty('instance creating')

        self.last_interaction = None
        self.pq, self.cq = None, None
        self.child_process = None
        self.reset_count = 0
        self.step_count = 0

        self.new_process()
        import threading as th
        self.lock = th.Lock()

    def timer_update(self):
        self.last_interaction = time.time()

    def is_occupied(self):
        if not self.occupied:
            return False
        else:
            if time.time() - self.last_interaction > 20 * 60:
                # if no interaction for more than 20 minutes
                self.pretty('no interaction for too long, self-releasing now. applying for a new id.')
                self.id = get_eid()  # apply for a new id.
                self.occupied = False
                self.pretty('self-released.')
                return False
            else:
                return True

    def occupy(self):
        self.lock.acquire()
        if not self.is_occupied():
            self.occupied = True
            self.id = get_eid()
            self.lock.release()
            return True
        else:
            self.lock.release()
            return False

    def release(self):
        self.lock.acquire()
        self.occupied = False
        self.id = get_eid()
        self.lock.release()

    # create a new RunEnv in a new process.
    def new_process(self):
        global plock
        self.timer_update()

        self.pq, self.cq = Queue(1), Queue(1)  # two queue needed
        # self.pc, self.cc = Pipe()

        self.child_process = Process(target=standalone_headless_isolated,
                                     args=(self.pq, self.cq, plock))
        self.child_process.daemon = True
        self.child_process.start()

        self.reset_count = 0  # how many times has this instance been reset()
        self.step_count = 0

        self.timer_update()
        return

    # send x to the process
    def send(self, x):
        return self.pq.put(x)

    # receive from the process.
    def recv(self):
        # receive and detect if we got any errors
        r = self.cq.get()

        # isinstance is dangerous, commented out
        # if isinstance(r,tuple):
        if r[0] == 'error':
            # read the exception string
            e = r[1]
            self.pretty('got exception')
            self.pretty(e)
            raise Exception(e)
        return r

    def reset(self):
        self.timer_update()
        if not self.is_alive():
            # if our process is dead for some reason
            self.pretty('process found dead on reset(). reloading.')
            self.kill()
            self.new_process()

        if self.reset_count > 50 or self.step_count > 10000:  # if reset for more than 50 times
            self.pretty(
                'environment has been resetted too much. memory leaks and other problems might present. reloading.')
            self.kill()
            self.new_process()

        self.reset_count += 1
        self.send(('reset',))
        r = self.recv()
        self.timer_update()
        return r

    def step(self, actions):
        self.timer_update()
        self.send(('step', actions,))
        r = self.recv()
        self.timer_update()
        self.step_count += 1
        return r

    def kill(self):
        if not self.is_alive():
            self.pretty('process already dead, no need for kill.')
        else:
            self.send(('exit',))
            self.pretty('waiting for join(timeout=5)...')
            while 1:
                self.child_process.join(timeout=5)
                if not self.is_alive():
                    break
                else:
                    self.pretty('process is not joining after 5s, still waiting...')
            self.pretty('process joined.')

    def __del__(self):
        self.pretty('__del__')
        self.kill()
        self.pretty('__del__ accomplished.')

    def is_alive(self):
        return self.child_process.is_alive()

    # pretty printing
    def pretty(self, s):
        print('(EI) {} '.format(self.id) + str(s))


# class that other classes acquires and releases EIs from.
class EIPool:  # Environment Instance Pool
    def pretty(self, s):
        print('(EIPool) ' + str(s))

    def __init__(self, n=1):
        import threading as th
        self.pretty('starting ' + str(n) + ' instance(s)...')
        self.pool = [EnvironmentInstance() for _ in range(n)]
        self.lock = th.Lock()

    def acq_env(self):
        self.lock.acquire()
        for environment in self.pool:
            if environment.occupy():  # successfully occupied an environment
                self.lock.release()
                return environment  # return the env instance

        self.lock.release()
        return False  # no available ei

    def rel_env(self, ei):
        self.lock.acquire()
        for e in self.pool:
            if e == ei:
                e.release()  # freed
        self.lock.release()

    def get_env_by_id(self, id):
        for e in self.pool:
            if e.id == id:
                return e
        return False

    def __del__(self):
        for e in self.pool:
            del e


# farm
# interface with EIPool via Eids.
# ! this class is a singleton. must be made thread-safe.
class Farm:
    def pretty(self, s):
        print('(Farm) ' + str(s))

    def __init__(self):
        import threading as th
        self.lock = th.Lock()

    def acq(self, n=None):
        self.renew_if_needed(n)
        result = self.eip.acq_env()  # thread-safe
        if not result:
            ret = False
        else:
            self.pretty('acq ' + str(result.id))
            ret = result.id
        return ret

    def rel(self, id):
        e = self.eip.get_env_by_id(id)
        if not e:
            self.pretty(str(id) + ' not found on rel(), might already be released')
        else:
            self.eip.rel_env(e)
            self.pretty('rel ' + str(id))

    def step(self, id, actions):
        e = self.eip.get_env_by_id(id)
        if not e:
            self.pretty(str(id) + ' not found on step(), might already be released')
            return False
        try:
            ordi = e.step(actions)
            return ordi
        except Exception as e:
            traceback.print_exc()
            raise e

    def reset(self, id):
        e = self.eip.get_env_by_id(id)
        if not e:
            self.pretty(str(id) + ' not found on reset(), might already be released')
            return False
        try:
            oo = e.reset()
            return oo
        except Exception as e:
            traceback.print_exc()
            raise e

    def renew_if_needed(self, n=None):
        self.lock.acquire()
        if not hasattr(self, 'eip'):
            self.pretty('renew because no eipool present')
            self._new(n)
        self.lock.release()

    def force_renew(self, n=None):
        self.lock.acquire()
        self.pretty('forced pool renew')

        if hasattr(self, 'eip'):  # if eip exists
            del self.eip
        self._new(n)
        self.lock.release()

    def _new(self, n=None):
        self.eip = EIPool(ncpu if n is None else n)


# expose the farm via Pyro4
def main():
    from utils.pyro_helper import pyro_expose
    pyro_expose(Farm, 20099, 'farm')


if __name__ == '__main__':
    main()
