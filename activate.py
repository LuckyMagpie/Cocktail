import cPickle
from scipy import signal
import librosa
import adaptfilt
from sys import argv

with open("netFile.pkl", "rb") as arch:
    net = cPickle.load(arch)


y,sr = librosa.load(argv[1], duration=10.0)
y2,sr2 = librosa.load("./sample/7/R7.wav", duration=10.0)


w = y
for i in xrange(int(argv[2])):
    r, s, n = net.activate(w)
    r = abs(int(round(r)))
    s = abs(int(round(s)))
    n = abs(n)
    print r, s, n

    for j in xrange(r):
        w = signal.wiener(w, mysize=s, noise=n)

    w= librosa.util.normalize(w)
    librosa.output.write_wav("net"+ str(i) +".wav", w, sr)
    out, w, coe = adaptfilt.nlmsru(y , w, 1, 0.0008)


