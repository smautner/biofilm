import dirtyopts as opts
import util.data as datautil
import numpy as np
import structout as so
from sklearn.metrics import  f1_score
from sklearn.model_selection import RandomizedSearchCV as RSCV
import searchspace as ss
from pprint import pprint
import matplotlib
matplotlib.use('module://matplotlib-sixel')

from matplotlib import pyplot as plt

optidoc='''
--method str ExtraTrees  whatever is specified in searchspace.py
--out str numpycompressdumpgoeshere_lol
--features str whereIdumpedMyFeatures
--o_iter int 2
--s_iter int 2
'''

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV as HRSCV
import copy

from lmz import *

from scipy.stats import randint, uniform
import scipy
NJOBS=1

def up(dis,make, typ):
    a,b = dis.args if typ == 'int' else (dis.args[0], sum(dis.args))
    low, high = np.mean([a,b]), b
    return make(low, high)

def down(dis,make, typ):
    a,b = dis.args if typ == 'int' else (dis.args[0], sum(dis.args))
    low, high = a, np.mean([a,b])
    return make(low,high)
    

from scipy.stats.mstats import pearsonr
def limitG(sc_v,al,k, orig):

    # categorical data:
    bestV = [v for s,v in sc_v[:10]]
    if isinstance(orig[k],list): 
        return  bestV
    
    # normal data:
    CUTLEN = len(sc_v)
    data = np.array(sc_v[:CUTLEN])
    if sum(data[:,1] ==  data[:,1][0]) == CUTLEN:
        return [bestV]

    corr = pearsonr(data[:,0],data[:,1])[0]
    #plt.scatter(data[:,1], data[:,0])
    #plt.show()
    #plt.close()
    
    if isinstance(al.dist, scipy.stats._discrete_distns.randint_gen):
        t= lambda x,y : al.dist(x,y), 'int'
    elif isinstance(al.dist, scipy.stats._continuous_distns.uniform_gen):
        t= lambda x,y : al.dist(x,y-x), 'float'
    else:
        assert False, f"type of {k} is not scipy randint or uniform"


    #if abs(corr)> .1:
    #    print(f"changing {k}: {corr}")
    # the plan is to ignore correlations < .1 
    if corr > 0.2:
        return up(al,*t)
    if corr < -0.2:
        return down(al,*t)
    return al

    
def limitparams(best,params, orig):
    best.sort(reverse=True, key=lambda x:x[0])
    nuparams={}
    for k,v in params.items():
        nuparams[k] = limitG([ (score,di[k]) for score, di in best ] ,params[k],k, orig)
    return nuparams


def li(di):
    return Zip(di['mean_test_score'], di['params'])

def fullsearch(X,Y,args, clf, params, niter):
    searcher = RSCV(clf,
                params,
                n_iter=niter,
                scoring='f1',
                n_jobs=NJOBS,
                cv=3,
                refit=False,
                random_state=None,
                error_score=np.nan,
                return_train_score=True)
    s = searcher.fit(X,Y).cv_results_
    return  li(s)


def showparm(params,param):
    for k,v in params.items():
        if isinstance(param[k],list):
            if  len(param[k]) > 1:
                print(k, v if isinstance(v,list)else (v.args))
        else:
            print(k, v if isinstance(v,list)else (v.args))

def halfsearch(X,Y,args, clf, params, niter):
    a,b,c,d = next(datautil.kfold(X,Y, n_splits=2)) 
    niter= int(niter/2)
    # Randomized search on hyper parameters
    searcher = RSCV(clf,
                params,
                n_iter=niter,
                scoring='f1',
                n_jobs=NJOBS,
                cv=3,
                refit=False,
                random_state=None,
                error_score=np.nan,
                return_train_score=True)
    r = searcher.fit(a,b).cv_results_
    s = searcher.fit(c,d).cv_results_
    return  li(r)+li(s)


def optimize(X,Y,x,y, args):
    clf, params = ss.classifiers[args.method]

    nuparams = copy.deepcopy(params)
    for i in range(args.o_iter):
        r = halfsearch(X,Y,args, clf, nuparams, 20) 
        nuparams = limitparams(r,nuparams, params)
        #showparm(nuparams,params)
        #print ("*"*30)

    searcher = RSCV(clf,
                nuparams,
                n_iter=args.s_iter*10,
                scoring='f1',
                n_jobs=NJOBS,
                cv=3,
                refit=True,
                random_state=None,
                error_score=np.nan,
                return_train_score=False)

    searcher.fit(X,Y)
    res = f1_score(y, searcher.predict(x))
    #print(res)
    return res



if __name__ == "__main__":
    args = opts.parse(optidoc)
    data = datautil.getfolds()
    res = [optimize(X,Y,x,y,args) for X,Y,x,y in data]
    print(np.mean(res), res)
    dat = opts.parse(datautil.datadoc)
    out = f'res/{args.o_iter}_{args.s_iter}_{dat.randinit}'
    np.savez_compressed(out,res)


# print avg in the end
# maybe train 10 on full / prune x3 and done


