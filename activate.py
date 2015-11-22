import cPickle
from scipy import signal
import librosa
import adaptfilt
from sys import argv

with open("netFile.pkl", "rb") as arch:
    net = cPickle.load(arch)


y,sr = librosa.load(argv[1], duration=10.0)

w = y
for i in xrange(int(argv[2])):
    r, s, n = net.activate(w)
    r = abs(int(round(r)))
    s = abs(int(round(s)))
    n = abs(n)
    print r, s, n


    #No hace falta iterar por que se reduce el rango dinamico
    #for j in xrange(r):
    #    w = signal.wiener(w, mysize=s, noise=n)

    w = signal.wiener(w, mysize=s, noise=n)

    #La normalizacion corrige variaciones importantes para el LMS como los maximos y minimos de la onda, es mejor desactivarla
    #w = librosa.util.normalize(w)
    librosa.output.write_wav("net"+ str(i) +".wav", w, sr)
    out, w, coe = adaptfilt.nlmsru(y , w, 1, 0.00001)


