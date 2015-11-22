import librosa
from scipy import signal
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.datasets import SupervisedDataSet
import cPickle
import wienerlayer
import adaptfilt
import backprop

ds = SupervisedDataSet(220500, 3)

y, sr = librosa.load("./sample/0/0.mp3", duration=10.0)
ds.addSample(y, [7,15,13])

y, sr = librosa.load("./sample/1/1.mp3", duration=10.0)
ds.addSample(y, [11,9,40])

y, sr = librosa.load("./sample/2/2.mp3", duration=10.0)
ds.addSample(y, [20,9,1])

y, sr = librosa.load("./sample/3/3.mp3", duration=10.0)
ds.addSample(y, [160,5,23])

y, sr = librosa.load("./sample/4/4.mp3", duration=10.0)
ds.addSample(y, [4,20,20])

y, sr = librosa.load("./sample/5/5.wav", duration=10.0)
ds.addSample(y, [10,76,70])

y, sr = librosa.load("./sample/6/6.wav", duration=10.0)
ds.addSample(y, [9,56,100])

y, sr = librosa.load("./sample/7/7.wav", duration=10.0)
ds.addSample(y, [20,45,80])


net = buildNetwork(220500, 80,  3)

trainer = backprop.BackpropTrainerWiener(net, ds)

for i in xrange(100):
	print trainer.train()

with open("netFile.pkl", "wb") as arch:
	cPickle.dump(net, arch, -1)
