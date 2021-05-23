import dirtyopts
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

datadoc='''
--infile str myNumpyDump
--randinit int -1
--folds int 5
--subsample int -1
--Z bool False
'''

def getfolds():
    args= dirtyopts.parse(datadoc).__dict__
    return loadfolds(**args)

def loadfolds(infile,randinit, folds, subsample, Z):
    
    d = np.load(infile)
    X,y = [d[f'arr_{x}'] for x in range(2)]

    if Z:
        X = StandardScaler().fit_transform(X)
    if subsample > 1:
        X,y= resample(X,y,replace=False,
                n_samples=subsample,
                random_state=randinit if randinit != -1 else None,
                stratify =y) 
    return kfold(X,y,folds,randseed=None if randinit == -1 else randinit)

def kfold(X, y, n_splits=5, randseed=None, shuffle=True):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=randseed)
    for train, test in kf.split(X, y):
        yield X[train], y[train], X[test], y[test]



