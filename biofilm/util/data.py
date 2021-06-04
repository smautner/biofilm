import dirtyopts
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from lmz import * 
datadoc='''
# theese are the options for reading data
--infile str myNumpyDump
--loader str  # optional path to a python file that introduces a load function


--randinit int 1337
--folds int 5
--foldselect int 0

--subsample int -1
--Z bool False

--featurefile str
--featurecount str -1 
'''

def getfold():
    args= dirtyopts.parse(datadoc).__dict__
    return loadfolds(**args)

def loadfolds(infile,randinit, folds, subsample, Z, loader,foldselect, featurefile, featurecount ):
    
    if not loader: 
        d = np.load(infile,allow_pickle=True)
        X,y = [d[f'arr_{x}'] for x in range(2)]
    else:
        eval(open(loader,'r').read()) 
        X,y = read(infile)
    
    if subsample > 1:
        X,y= resample(X,y,replace=False,
                n_samples=subsample,
                random_state=randinit,
                stratify =y) 

    if featurefile:
        if featurecount > 0: # size constrained list
            ft_quality = np.load(featurefile, allow_pickle=True)['arr_1']
            
            want=np.argsort(feature_quality)[-featurecount:]
            X=X[:,want]
            
        else: # default list 
            ft = np.load(featurefile)['arr_0']
            X=X[:,ft%2==1] # works with 0/1 and False/True

    if Z:
        X = StandardScaler().fit_transform(X)

    return  iterselect( kfold(X,y,folds,randseed=randinit), foldselect)

def kfold(X, y, n_splits=5, randseed=None, shuffle=True):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=randseed)
    for train,test in  kf.split(X, y):
        yield X[train], y[train], X[test], y[test]


