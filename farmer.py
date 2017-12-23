# farmer.py
# Adapted from https://github.com/ctmakro/stanford-osrl

import time

# connector to the farms
from utils.pyro_helper import pyro_connect


class FarmList:
    def __init__(self):
        self.list = []

    def generate(self):
        farmport = 20099

        def addressify(farmaddr, port):
            return farmaddr + ':' + str(port)

        addresses_from_farmlist = [addressify(farm[0], farmport) for farm in self.list]
        capacities_from_farmlist = [farm[1] for farm in self.list]
        failures_from_farmlist = [0 for _ in range(len(capacities_from_farmlist))]

        return addresses_from_farmlist, capacities_from_farmlist, failures_from_farmlist

    def push(self, address, capacity):
        self.list.append((address, capacity))


farm_list = FarmList()


def reload_address():
    global addresses, capacities, failures

    g = {'nothing': []}
    with open('farmlist.py', 'r') as f:
        farmlist_py = f.read()
    exec (farmlist_py, g)
    farmlist_base = g['farmlist_base']

    farm_list.list = []
    for item in farmlist_base:
        farm_list.push(item[0], item[1])

    addresses, capacities, failures = farm_list.generate()


reload_address()


class RemoteEnv:
    def pretty(self, s):
        print('(emoteEnv) {} '.format(self.id) + str(s))

    def __init__(self, farm_proxy, id):  # fp = farm proxy
        self.farm_proxy = farm_proxy
        self.id = id

    def reset(self):
        return self.farm_proxy.reset(self.id)

    def step(self, actions):
        ret = self.farm_proxy.step(self.id, actions)
        if not ret:
            self.pretty('env not found on farm side, might been released.')
            raise Exception('env not found on farm side, might been released.')
        return ret

    def rel(self):
        count = 0
        while True:  # releasing is important, so
            try:
                count += 1
                self.farm_proxy.rel(self.id)
                break
            except Exception as e:
                self.pretty('exception caught on rel()')
                self.pretty(e)
                time.sleep(3)
                if count > 5:
                    self.pretty('failed to rel().')
                    break
                pass

        self.farm_proxy._pyroRelease()

    def __del__(self):
        self.rel()


class Farmer:
    def reload_address(self):
        self.pretty('reloading farm list...')
        reload_address()

    def pretty(self, s):
        print('(farmer) ' + str(s))

    def __init__(self):
        for idx, address in enumerate(addresses):
            fp = pyro_connect(address, 'farm')
            self.pretty('forced renewing... ' + address)
            try:
                fp.force_renew(capacities[idx])
                self.pretty('fp.forcerenew() success on ' + address)
            except Exception as e:
                self.pretty('fp.forcerenew() failed on ' + address)
                self.pretty(e)
                fp._pyroRelease()
                continue
            fp._pyroRelease()

    # find non-occupied instances from all available farms
    def acq_env(self):
        ret = False

        import random  # randomly sample to achieve load averaging
        indexes = list(range(len(addresses)))
        random.shuffle(indexes)

        for idx in indexes:
            time.sleep(0.1)
            address = addresses[idx]
            capacity = capacities[idx]

            if failures[idx] > 0:
                # wait for a few more rounds upon failure,
                # to minimize overhead on querying busy instances
                failures[idx] -= 1
                continue
            else:
                fp = pyro_connect(address, 'farm')
                try:
                    result = fp.acq(capacity)
                except Exception as e:
                    self.pretty('fp.acq() failed on ' + address)
                    self.pretty(e)

                    fp._pyroRelease()
                    failures[idx] += 4
                    continue
                else:
                    if not result:  # no free ei
                        fp._pyroRelease()  # destroy proxy
                        failures[idx] += 4
                        continue
                    else:  # result is an id
                        eid = result
                        renv = RemoteEnv(fp, eid)  # build remoteEnv around the proxy
                        self.pretty('got one on {} id:{}'.format(address, eid))
                        ret = renv
                        break

        # ret is False if none of the farms has free ei
        return ret
