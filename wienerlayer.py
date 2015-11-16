from pybrain.structure.modules.neuronlayer import NeuronLayer
from scipy import signal


class WienerLayer(NeuronLayer):

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = signal.wiener(inbuf)

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:] = outerr
