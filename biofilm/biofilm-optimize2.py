import dirtyopts as opts
import util.data as datautil
import numpy as np
import structout as so
from sklearn.metrics import  f1_score
from sklearn.model_selection import RandomizedSearchCV as RSCV
import searchspace as ss

optidoc='''
--method str xtratrees  whatever is specified in searchspace.py
--out str numpycompressdumpgoeshere_lol
--features str whereIdumpedMyFeatures
'''

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV as HRSCV

def optimize(X,Y,x,y, args):
    clf, params = ss.classifiers[args.method]
    # Randomized search on hyper parameters
    searcher = RSCV(clf,
                params,
                n_iter=30,
                scoring='f1',
                n_jobs=1,
                cv=3,
                random_state=42,
                error_score=np.nan,
                return_train_score=True)
    searcher.fit(X, Y)
    #print (searcher.__dict__)
    print(f1_score(y, searcher.predict(x)))


def optimize2(X,Y,x,y, args):
    clf, params = ss.classifiers[args.method]
    # Randomized search on hyper parameters
    searcher = HRSCV(clf,
                params,
                scoring='f1',
                n_jobs=1,
                cv=3,
                random_state=42,
                error_score=np.nan,
                return_train_score=True)
    searcher.fit(X, Y)
    # todo: print mean of sorted train scores.. 
    print(f1_score(y, searcher.predict(x)))

if __name__ == "__main__":
    args = opts.parse(optidoc)
    data = datautil.getfolds()
    name = 'lol'
    np.savez_compressed(
            args.out, [optimize2(X,Y,x,y,args) for X,Y,x,y in data])




