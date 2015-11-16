from scipy import dot, argmax
from random import shuffle
from pybrain.supervised.trainers import Trainer
from pybrain.utilities import fListToString
from pybrain.auxiliary import GradientDescent


class BackpropTrainerWiener(Trainer):

    def __init__(self, module, dataset=None, learningrate=0.01, lrdecay=1.0,
                 momentum=0., verbose=False, batchlearning=False,
                 weightdecay=0.):
        Trainer.__init__(self, module)
        self.setData(dataset)
        self.verbose = verbose
        self.batchlearning = batchlearning
        self.weightdecay = weightdecay
        self.epoch = 0
        self.totalepochs = 0
        # set up gradient descender
        self.descent = GradientDescent()
        self.descent.alpha = learningrate
        self.descent.momentum = momentum
        self.descent.alphadecay = lrdecay
        self.descent.init(module.params)

    def train(self):
        assert len(self.ds) > 0, "Dataset cannot be empty."
        self.module.resetDerivatives()
        errors = 0
        ponderation = 0.
        shuffledSequences = []
        for seq in self.ds._provideSequences():
            shuffledSequences.append(seq)
        shuffle(shuffledSequences)
        for seq in shuffledSequences:
            e, p = self._calcDerivs(seq)
            errors += e
            ponderation += p
            if not self.batchlearning:
                gradient = self.module.derivs - self.weightdecay * self.module.params
                new = self.descent(gradient, errors)
                if new is not None:
                    self.module.params[:] = new
                self.module.resetDerivatives()

        if self.verbose:
            print "Total error:", errors / ponderation
        if self.batchlearning:
            self.module._setParameters(self.descent(self.module.derivs))
        self.epoch += 1
        self.totalepochs += 1
        return errors / ponderation


    def _calcDerivs(self, seq):
        self.module.reset()
        for sample in seq:
            self.module.activate(sample[0])
        error = 0
        ponderation = 0.
        for offset, sample in reversed(list(enumerate(seq))):
            target = sample[1]
            outerr = target - self.module.outputbuffer[offset]
            if len(sample) > 2:
                importance = sample[2]
                error += 0.5 * dot(importance, outerr ** 2)
                ponderation += sum(importance)
                self.module.backActivate(outerr * importance)
            else:
                error += 0.5 * sum(outerr ** 2)
                ponderation += len(target)
                str(outerr)
                self.module.backActivate(outerr)

        return error, ponderation

    def _checkGradient(self, dataset=None, silent=False):
        if dataset:
            self.setData(dataset)
        res = []
        for seq in self.ds._provideSequences():
            self.module.resetDerivatives()
            self._calcDerivs(seq)
            e = 1e-6
            analyticalDerivs = self.module.derivs.copy()
            numericalDerivs = []
            for p in range(self.module.paramdim):
                storedoldval = self.module.params[p]
                self.module.params[p] += e
                righterror, dummy = self._calcDerivs(seq)
                self.module.params[p] -= 2 * e
                lefterror, dummy = self._calcDerivs(seq)
                approxderiv = (righterror - lefterror) / (2 * e)
                self.module.params[p] = storedoldval
                numericalDerivs.append(approxderiv)
            r = zip(analyticalDerivs, numericalDerivs)
            res.append(r)
            if not silent:
                print r
        return res

    def testOnData(self, dataset=None, verbose=False):
        if dataset == None:
            dataset = self.ds
        dataset.reset()
        if verbose:
            print '\nTesting on data:'
        errors = []
        importances = []
        ponderatedErrors = []
        for seq in dataset._provideSequences():
            self.module.reset()
            e, i = dataset._evaluateSequence(self.module.activate, seq, verbose)
            importances.append(i)
            errors.append(e)
            ponderatedErrors.append(e / i)
        if verbose:
            print 'All errors:', ponderatedErrors
        assert sum(importances) > 0
        avgErr = sum(errors) / sum(importances)
        if verbose:
            print 'Average error:', avgErr
            print ('Max error:', max(ponderatedErrors), 'Median error:',
                   sorted(ponderatedErrors)[len(errors) / 2])
        return avgErr

    def testOnClassData(self, dataset=None, verbose=False,
                        return_targets=False):
        if dataset == None:
            dataset = self.ds
        dataset.reset()
        out = []
        targ = []
        for seq in dataset._provideSequences():
            self.module.reset()
            for input, target in seq:
                res = self.module.activate(input)
                out.append(argmax(res))
                targ.append(argmax(target))
        if return_targets:
            return out, targ
        else:
            return out

    def trainUntilConvergence(self, dataset=None, maxEpochs=None, verbose=None,
                              continueEpochs=10, validationProportion=0.25):
        epochs = 0
        if dataset == None:
            dataset = self.ds
        if verbose == None:
            verbose = self.verbose
        trainingData, validationData = (
            dataset.splitWithProportion(1 - validationProportion))
        if not (len(trainingData) > 0 and len(validationData)):
            raise ValueError("Provided dataset too small to be split into training " +
                             "and validation sets with proportion " + str(validationProportion))
        self.ds = trainingData
        bestweights = self.module.params.copy()
        bestverr = self.testOnData(validationData)
        trainingErrors = []
        validationErrors = [bestverr]
        while True:
            trainingErrors.append(self.train())
            validationErrors.append(self.testOnData(validationData))
            if epochs == 0 or validationErrors[-1] < bestverr:
                bestverr = validationErrors[-1]
                bestweights = self.module.params.copy()

            if maxEpochs != None and epochs >= maxEpochs:
                self.module.params[:] = bestweights
                break
            epochs += 1

            if len(validationErrors) >= continueEpochs * 2:
                old = validationErrors[-continueEpochs * 2:-continueEpochs]
                new = validationErrors[-continueEpochs:]
                if min(new) > max(old):
                    self.module.params[:] = bestweights
                    break
        trainingErrors.append(self.testOnData(trainingData))
        self.ds = dataset
        if verbose:
            print 'train-errors:', fListToString(trainingErrors, 6)
            print 'valid-errors:', fListToString(validationErrors, 6)
        return trainingErrors, validationErrors
