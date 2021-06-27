import dirtyopts
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from lmz import * 
import scipy.sparse as sparse
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
--featurecount int -1 
'''

def getfold():
    args= dirtyopts.parse(datadoc).__dict__
    return loadfolds(**args)

def loadfolds(infile,loader,randinit, folds,foldselect,  subsample, Z, featurefile, featurecount ):
    
    if not loader: 
        d = np.load(infile,allow_pickle=True)
        X,y  = [d[f'arr_{x}'] for x in range(2)]
        instances = np.array(Range(X.shape[0]))
        features = np.array(Range(X.shape[1]))
    
    else:
        scope = {}
        exec(open(loader,'r').read(), scope) 
        X,y, features, instances = scope['read'](infile)
    
    if featurefile != '':
        if featurecount > 0: # size constrained list
            ft_quality = np.load(featurefile, allow_pickle=True)['arr_1']
            
            want=np.argsort(feature_quality)[-featurecount:]

            X=X[:,want]
            features=features[:,want]
            
        else: # default list 
            ft = np.load(featurefile)['arr_0']
            X=X[:,ft%2==1] # works with 0/1 and False/True
            features=features[ft%2==1] # works with 0/1 and False/True



    if subsample > 1:
        X,y, instances = resample(X,y,instances, replace=False,
                n_samples=subsample,
                random_state=randinit,
                stratify =y) 
    if Z:
        X = StandardScaler(with_mean= type(X) != sparse.csr_matrix ).fit_transform(X )

    return iterselect( kfold(X,y,n_splits = folds,randseed=randinit, feature_names = features, instance_names = instances),  foldselect)

def kfold(X, y, n_splits=5, randseed=None, shuffle=True, feature_names=None, instance_names=None):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=randseed)
    for train,test in  kf.split(X, y):
        yield (X[train], y[train], X[test], y[test]),  feature_names, instance_names[test] 


