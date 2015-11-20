import librosa
from scipy import signal
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.datasets import SupervisedDataSet
import cPickle
import wienerlayer
import adaptfilt
import backprop

ds = SupervisedDataSet(220500, 220500)
for i in xrange(5):
	if (i<5):
		y, sr = librosa.load("./sample/" +str(i)+ "/" +str(i)+ ".mp3", duration=10.0)
		y2, sr2 = librosa.load("./sample/" +str(i)+ "/R" +str(i)+ ".mp3", duration=10.0)
		ds.addSample(y, y2)

	else:
		y, sr = librosa.load("./sample/" +str(i)+ "/" +str(i)+ ".wav", duration=10.0)
		y2, sr2 = librosa.load("./sample/" +str(i)+ "/R" +str(i)+ ".wav", duration=10.0)
		ds.addSample(y, y2)

net = buildNetwork(220500, 35, 30, 25, 20, 15, 13, 6, 3, 220500, hiddenclass=TanhLayer,outclass=wienerlayer.WienerLayer)

trainer = backprop.BackpropTrainerWiener(net, ds)

for i in xrange(200):
	print trainer.train()

with open("netFile.pkl", "wb") as arch:
	cPickle.dump(net, arch, -1)
