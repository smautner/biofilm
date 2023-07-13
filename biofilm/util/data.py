import dirtyopts
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from lmz import iterselect, Range
import scipy.sparse as sparse
from ubergauss import tools
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

--instancegroups str   # a jsonfile containing a dictionary instance_name -> group name
'''


def getargs():
    return dirtyopts.parse(datadoc)

def getfold():
    args= dirtyopts.parse(datadoc).__dict__
    return loadfolds(**args)

def getgroups(groups, instance_names):
    groupdict = tools.jloadfile(groups)
    grouplist = [groupdict[str(i)] for i in instance_names]
    groupintlist = tools.labelsToIntList(grouplist)[0]
    return groupintlist


from sklearn.model_selection import  BaseCrossValidator

def contains(iter, element):
    for e in iter:
        if e == element:
            return True
    return False

class groupedCV(BaseCrossValidator):
    def __init__(self, n_splits):
        self.n_splits = n_splits
    def get_n_splits(self, X= None, y= None, groups = None):
        return self.n_splits

    # def arin(self,groupindex, testgrps):
    #     return np.array([ contains(testgrps, a) for a in groupindex])

    def arin_index(self,groupindex, testgrps):
        return np.array([i for i,a in enumerate(groupindex) if contains(testgrps,a)])

    def _iter_test_indices(self, X,y,groups):
        groups = np.array(groups)
        z = np.unique(groups)
        np.random.shuffle(z)
        if self.n_splits > 1:
            for testgroups in np.array_split(z, self.n_splits):
                res =  self.arin_index(groups, testgroups)
                yield res
        else:
            test = np.array_split(z, 3)[0]
            yield self.arin_index(groups, test)

    # def _iter_test_mask(self, X,y,groups):
    #     groups = np.array(groups)
    #     z = np.unique(groups)
    #     np.random.shuffle(z)
    #     if self.n_splits > 1:
    #         for testgroups in np.split(z, self.n_splits):
    #             res =  groups in testgroups
    #             print(f"{ res=}")
    #             yield res
    #     else:
    #         test = np.split(z, self.n_splits)
    #         yield groups in test


def loadfolds(infile=None,loader=None,randinit=None, folds=None,
              foldselect=None, subsample=None, Z=None, featurefile=None, featurecount=None, instancegroups = ''):
    if not loader:
        # assume data as saved via ubergauss.tools.ndumpfile
        raw = tools.nloadfile(infile)
        X  = raw[0]
        if len(raw) < 3:
            raw.append( np.array(Range(X.shape[1])) )
        if len(raw) < 4:
            raw.append( np.array(Range(X.shape[0])) )
        """
        d = np.load(infile,allow_pickle=True)
        X,y  = [d[f'arr_{x}'] for x in range(2)]
        print(f" {X.shape=} {y.shape=}")
        instances = np.array(Range(X.shape[0]))
        features = np.array(Range(X.shape[1]))
        """

        X,y, features, instances = raw
    else:
        scope = {}
        exec(open(loader,'r').read(), scope)
        X,y, features, instances = scope['read'](infile)

    assert X.shape[1] == len(features), f'{X.shape[1]=} {len(features)=}'

    if featurefile != '':
        featurefile += '.npz'
        if featurecount > 0: # size constrained list
            ft_quality = np.load(featurefile, allow_pickle=True)['arr_1']
            want=np.argsort(ft_quality)[-featurecount:]
            X=X[:,want]
            features=features[:,want]

        else: # default list
            ft = np.load(featurefile)['arr_0']
            X=X[:,ft%2==1] # works with 0/1 and False/True
            features=[ f for f, ok in zip(features,ft%2==1) if ok]



    if subsample > 1 and X.shape[0]>subsample:
        X,y, instances = resample(X,y,instances, replace=False,
                n_samples=subsample,
                random_state=randinit,
                stratify =y)
    if Z:
        X = StandardScaler(with_mean= type(X) != sparse.csr_matrix ).fit_transform(X )

    if folds > 1:
        return iterselect(
                kfold(X,y,n_splits = folds,randseed=randinit,
                    feature_names = features, instance_names = instances, groups=instancegroups),
                foldselect)
    else:
        return (X,y,np.array([]),np.array([])),features, {f'train':instances,
                                                          f'test':[]}


def kfold(X, y, n_splits=5, randseed=None, shuffle=True, feature_names=None, instance_names=None, groups = ''):

    if not groups:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=randseed).split(X,y)
    else:
        groupintlist = getgroups(groups, instance_names)
        # kf = groupedCV(n_splits=n_splits, shuffle=shuffle, random_state=randseed).split(X, y, groupintlist )
        kf = groupedCV(n_splits=n_splits).split(X, y, groupintlist )

    for train,test in  kf:
        yield (X[train], y[train], X[test], y[test]),  feature_names, {f'train':instance_names[train],
                                                                       f'test':instance_names[test]}


