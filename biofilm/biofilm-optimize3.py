import dirtyopts as opts
import util.data as datautil
import numpy as np
import structout as so
from sklearn.metrics import  f1_score
from sklearn.model_selection import RandomizedSearchCV as RSCV
import searchspace as ss
from skopt import BayesSearchCV
optidoc='''
--method str ExtraTrees  whatever is specified in searchspace.py
--out str numpycompressdumpgoeshere_lol
--features str whereIdumpedMyFeatures
'''

# DEV VERSION WORKS :) 

def optimize(X,Y,x,y, args):
    clf, params = ss.classifiers[args.method]
    # Randomized search on hyper parameters
    #https://scikit-optimize.github.io/stable/auto_examples/sklearn-gridsearchcv-replacement.html
    searcher = BayesSearchCV(clf,params,n_iter=30,cv=3)
    searcher.fit(X, Y )
    #print (searcher.__dict__)
    print(f1_score(y, searcher.predict(x)))

if __name__ == "__main__":
    args = opts.parse(optidoc)
    data = datautil.getfolds()
    name = 'lol'
    np.savez_compressed(
            args.out, [optimize(X,Y,x,y,args) for X,Y,x,y in data])




