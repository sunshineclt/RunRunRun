import tensorflow as tf


class TensorFlowFunction(object):
    def __init__(self, sess, inputs, outputs):
        self.sess = sess
        self._inputs = inputs
        self._outputs = outputs

    def __call__(self, *args, **kwargs):
        feeds = {}
        for (argpos, arg) in enumerate(args):
            feeds[self._inputs[argpos]] = arg
        outputs_list = self.sess.run(self._outputs, feeds)
        return tuple(outputs_list)
