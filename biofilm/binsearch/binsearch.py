import numpy as np
import structout as so
from sklearn.metrics import  f1_score
from sklearn.model_selection import RandomizedSearchCV as RSCV
import searchspace as ss
from pprint import pprint
import copy
from lmz import *
import scipy
from sklearn.model_selection import StratifiedKFold
from scipy.stats.mstats import pearsonr
import binsearch.limit as limit

class binsearch(RSCV): 
    def fit(self, X, y=None, *, groups=None, **fit_params):
        assert self.n_iter >=25, "n_iter should be at least 25"
        # we want to change theese:
        self.binlog = 0,{}
        self.param_distributions = self.binsearch(X,y,self.param_distributions) 
        super().fit(X,y,groups=groups,**fit_params)

    def binsearch(self,x,y, params):
        nuparams = copy.deepcopy(params)
        for i in range(2):
        #while self.n_iter > 15:
            r = self.halfsearch(x,y, nuparams, 20)
            mapa = max(r, key = lambda x:x[0])
            if mapa[0] > self.binlog[0]:
                self.binlog = mapa
            nuparams , contchrchange = self.limitparams(r,nuparams, params)
            self.n_iter -=10 
            #print ("numchanges", contchrchange) 
            #pparm(nuparams)
        return nuparams

    def halfsearch(self,X,Y, params, niter):
        f1,f2  =next(StratifiedKFold(n_splits=2 , shuffle=True).split(X,Y))
        a,b,c,d = X[f1],Y[f1],X[f2],Y[f2]
        niter= int(niter/2)
        # Randomized search on hyper parameters
        searcher = RSCV(self.estimator,
                    params,
                    n_iter=niter,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs,
                    cv=self.cv,
                    refit=False,
                    random_state=None,
                    error_score=np.nan,
                    return_train_score=True)
        r = searcher.fit(a,b).cv_results_
        s = searcher.fit(c,d).cv_results_
        li = lambda di: Zip(di['mean_test_score'], di['params'])
        return  li(r)+li(s)

    def limitparams(self,best,curparams, allparams):
        best.sort(reverse=True, key=lambda x:x[0])
        nuparams={}
        ch = 0
        for k,v in curparams.items():
            nuparams[k],grr= limit.limit([ (score,di[k]) for score, di in best ] ,curparams[k],k, allparams[k])
            ch += grr
        return nuparams,ch


def pparm(params):
    print ("^"*80)
    for k,v in params.items():
        if isinstance(v,list):
            if  not all( [x == v[0] for x in v]  ):
                print(k, v if isinstance(v,list) else (v.args))
        else:
            print(k, v if isinstance(v,list) else (v.args))
    pprint(params)

