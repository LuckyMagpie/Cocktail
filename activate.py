import cPickle
from sys import argv

with open("netFile.pkl", "rb") as arch:
    net = cPickle.load(arch)

script, sound, num = argv
