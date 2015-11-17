import librosa
from scipy import signal
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.datasets import SupervisedDataSet
import cPickle
import wienerlayer
import adaptfilt
import backprop

y, sr = librosa.load("./sample/asd.wav", duration=15.0)
y2, sr2 = librosa.load("./sample/asd1.wav", duration=15.0)

net = buildNetwork(len(y), 35, 30, 25, 20, 15, 13, 6, 3, len(y2), hiddenclass=TanhLayer,outclass=wienerlayer.WienerLayer)
ds = SupervisedDataSet(len(y), len(y2))

ds.addSample(y, y2)
# ...

trainer = backprop.BackpropTrainerWiener(net, ds)

for i in xrange(1):
    print trainer.train()

with open("netFile.pkl", "wb") as arch:
    cPickle.dump(net, arch, -1)
