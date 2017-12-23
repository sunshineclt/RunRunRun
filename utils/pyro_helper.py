# Pyro4 library helper
# Adapted from https://github.com/ctmakro/stanford-osrl

import Pyro4 as p4

p4.config.HOST = '0.0.0.0'
p4.config.COMMTIMEOUT = 1200.0  # 20 min timeout
p4.config.THREADPOOL_SIZE = 1000


def pyro_connect(address, name):
    uri = 'PYRO:' + name + '@' + address
    return p4.Proxy(uri)


def pyro_expose(c, port, name):
    def stop():
        print('stop() called')
        import os
        os._exit(1)

    from triggerbox import TriggerBox
    tb = TriggerBox(name + ' server on ' + str(port),
                    ['stop server'],
                    [stop])

    c = p4.behavior(instance_mode='single')(c)
    exposed = p4.expose(c)
    p4.Daemon.serveSimple(
        {exposed: name},
        ns=False,
        port=port)

    daemon = p4.Daemon()
    daemon.register(c, name)
    daemon.requestLoop()