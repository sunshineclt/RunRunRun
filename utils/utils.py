import tensorflow as tf


class TensorFlowFunction(object):
    def __init__(self, inputs, outputs):
        self._inputs = inputs
        self._outputs = outputs

    def __call__(self, *args, **kwargs):
        feeds = {}
        for (argpos, arg) in enumerate(args):
            feeds[self._inputs[argpos]] = arg
        outputs_list = tf.get_default_session().run(self._outputs, feeds)
        return tuple(outputs_list)
