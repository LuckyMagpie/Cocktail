import cPickle
import librosa
from sys import argv

with open("netFile.pkl", "rb") as arch:
    net = cPickle.load(arch)


y,sr = librosa.load(argv[1], duration=10.0)

y = net.activate(y)
librosa.output.write_wav("net.wav", y, sr)

