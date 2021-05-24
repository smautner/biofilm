import dirtyopts as opts
import util.data as datautil
import numpy as np
import structout as so
from sklearn.metrics import  f1_score
from sklearn.model_selection import RandomizedSearchCV as RSCV
import searchspace as ss
from pprint import pprint
optidoc='''
--method str ExtraTrees  whatever is specified in searchspace.py
--out str numpycompressdumpgoeshere_lol
--features str whereIdumpedMyFeatures
'''

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV as HRSCV


from lmz import *

def index(ra,v):
    if isinstance(ra,list):
        return ra.index(v)
    else:
        return np.where(ra == v)[0][0]

def pick(rang, mi,ma): 
    if mi == ma:
        return [mi]
    if mi in rang:
        low = index(rang,mi)
    else:
        low = 0
    if ma in rang:
        high = index(rang,ma)
    else:
        high = None
    res=  rang[low:high]
    #print (rang,mi,ma, res, low, high)
    return res


def limit(sc_v,al,k):
    if len(al) < 5: 
        # lets assume categorical 
        # distribute p proportionally
        return [v for s,v in sc_v[:5]]
    z = [(v,s) for s,v in sc_v[:5]]
    z.sort()
    z=np.array(z)
    #if k == 'n_estimators': print (z)
    ind =np.argsort(-z[:,1])[0]
    v1,v2,v3,v4,v5 = z[:,0]

    if ind == 2:
        return pick(al,v2,v4)

    if ind == 0: 
        return pick(al,v1-v5,v3)
    if ind == 4:  
        return pick(al,v3,v5+v1)

    if ind == 1: 
        return pick(al,v1,v3)
    if ind == 3:
        return pick(al,v3,v5)
    

from scipy.stats.mstats import pearsonr
def limitG(sc_v,al,k, orig):
    if len(orig[k]) < 5: 
        # lets assume categorical 
        # distribute p proportionally
        return [v for s,v in sc_v[:5]]

    bestV = sc_v[0][1]
    CUTLEN = 6
    data = np.array(sc_v[:CUTLEN])
    if sum(data[:,1] ==  data[:,1][0]) == CUTLEN:
        return [bestV]

    #print(data)
    corr = pearsonr(data[:,0],data[:,1])[0]
    
    if corr > 0.15:
        return pick(al,bestV, max(al)) 
    if corr < -0.15:
        return pick(al,min(al), bestV) 
    return [bestV]
    
def limitparams(best,params, orig):
    # best is a list: score:dict
    # params is a dict:lists
    best.sort(reverse=True, key=lambda x:x[0])
    nuparams={}
    for k,v in params.items():
        nuparams[k] = limitG([ (score,di[k]) for score, di in best ] ,params[k],k, orig)
        #print (k, nuparams[k], params[k])
    return nuparams



def li(di):
    return Zip(di['mean_test_score'], di['params'])

def halfsearch(X,Y,args, clf, params):
    a,b,c,d = next(datautil.kfold(X,Y))
    # Randomized search on hyper parameters
    searcher = RSCV(clf,
                params,
                n_iter=10,
                scoring='f1',
                n_jobs=1,
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
    r = halfsearch(X,Y,args, clf, params)
    nuparams = limitparams(r,params, params)
    r+= halfsearch(X,Y,args, clf, nuparams)
    nuparams = limitparams(r,nuparams,params)
    pprint (nuparams)
    #pprint (r)
    searcher = RSCV(clf,
                nuparams,
                n_iter=10,
                scoring='f1',
                n_jobs=1,
                cv=3,
                refit=True,
                random_state=None,
                error_score=np.nan,
                return_train_score=False)
    searcher.fit(X,Y)
    print(f1_score(y, searcher.predict(x)))
    #exit()

if __name__ == "__main__":
    args = opts.parse(optidoc)
    data = datautil.getfolds()
    np.savez_compressed(
            args.out, [optimize(X,Y,x,y,args) for X,Y,x,y in data])

# print avg in the end
# maybe train 10 on full / prune x3 and done


