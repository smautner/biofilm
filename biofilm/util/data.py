import dirtyopts
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

datadoc='''
# theese are the options for reading data
--infile str myNumpyDump
--loader str  # optional path to a python file that introduces a load function
--randinit int 1337
--folds int 5
--subsample int -1
--Z bool False
--foldselect int -1
--featurefile str
# TODO specipy a single fold  :) 
'''

def getfolds():
    args= dirtyopts.parse(datadoc).__dict__
    return loadfolds(**args)

def loadfolds(infile,randinit, folds, subsample, Z, loader):
    
    if not loader: 
        d = np.load(infile)
        X,y = [d[f'arr_{x}'] for x in range(2)]
    else:
        eval(open(loader,'r').read()) 
        X,y = read(infile)
    
    if featurefile:
        ft = np.load(featurefile)['arr_0']
        X=X[:,ft%2==1] # works with 0/1 and False/True

    if Z:
        X = StandardScaler().fit_transform(X)
    if subsample > 1:
        X,y= resample(X,y,replace=False,
                n_samples=subsample,
                random_state=randinit if randinit != -1 else None,
                stratify =y) 
    return kfold(X,y,folds,randseed=None if randinit == -1 else randinit)

def kfold(X, y, n_splits=5, randseed=None, shuffle=True, foldselect =-1):

    kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=randseed)
    if foldselect == -1:
        for train, test in kf.split(X, y):
            yield X[train], y[train], X[test], y[test]
    else:
        print("NOT IMPLEMENTED")


