import dirtyopts as opts
from lmz import *

import biofilm.util.data as datautil
import numpy as np
from sklearn.linear_model import SGDClassifier as sgd, LassoCV
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
import matplotlib
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV
import structout as so
import random
import ubergauss
from scipy.stats import spearmanr
matplotlib.use('module://matplotlib-sixel')
import biofilm.util.draw as draw
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from statsmodels.stats.outliers_influence import variance_inflation_factor


featdoc=''' 
# options for feature selection:
--method str lasso  svm all corr variance logistic relaxedlasso  VarianceInflation
--out str numpycompressdumpgoeshere
--plot bool False
--svmparamrange float+ 0.01 0.15 0.001

'''
def relaxedlasso(X,Y,x,y,args):
    print("RELAXED LASSO NOT IMPLEMENTD ") # TODO 

def lasso(X,Y,x,y,args):
    model = LassoCV(n_alphas = 100).fit(X,Y)
    if args.plot:
        testscore, cutoff = max([(f1_score(Y,model.predict(X) > t),t) for t in np.arange(.3,.7,.001)])
        print ('score: %.2f alpha: %.4f  features: %d  ERRORPATH, FEATCOUNT' % ( f1_score(y,model.predict(x)>cutoff), model.alpha_, (model.coef_>0.0001).sum()))
        draw.lasso(model,X,Y)
    quality = abs(model.coef_)
    print(f" quality{quality}")
    res =  quality > 0.0001
    so.lprint(res.astype(np.int64))
    return res, quality

def logistic(X,Y,x,y,args):
    model = LogisticRegressionCV(Cs=10, penalty = 'l1',n_jobs = -1, max_iter = 1000, solver ='liblinear').fit(X,Y)
    # TODO test this, one may want to change Cs...  also liblinear / saga
    quality = abs(model.coef_.ravel())
    res =  quality > 0.0001
    so.lprint(res.astype(np.int64))
    return res, quality


def svm(X,Y,x,y,args): 
    clf = LinearSVC(class_weight = 'balanced', max_iter=10000)
    param_dist = {"C": np.arange(*args.svmparamrange) , 'penalty':['l1'],'dual':[False]}
    search = GridSearchCV(clf,param_dist, n_jobs=10, scoring='f1')
    search.fit(X, Y)

    model = search.best_estimator_
    if args.plot:
        print ("numft %d  C %.3f  testscore %.2f SCOREPATH:" % 
                ((model.coef_>0.0001 ).sum(),model.C,f1_score(y,model.predict(x))))

        err = search.cv_results_["mean_test_score"]
        so.lprint(err)

    quality = abs(model.coef_)
    res = ( quality > 0.0001).squeeze()
    so.lprint(res.astype(np.int64))
    return res, quality

def autothresh(arr):
    arr=abs(arr)
    cpy = np.array(arr)
    cpy.sort()
    rr = ubergauss.between_gaussians(cpy)
    return arr >= cpy[rr]

def variance(X,Y,x,y,args):
    var = np.var(X, axis = 0)
    res = (autothresh(var))
    so.lprint(res.astype(np.int64))
    if args.plot:
        # TODO 
        # sort var 
        # lprint 
        pass

    return res, var

def corr(X,Y,x,y,args):
    cor = abs(np.array([spearmanr(X[:,column],Y)[0] for column in range(X.shape[1])]))
    res = (autothresh(cor))
    so.lprint(res.astype(np.int64))
    if args.plot:
        # sort corr  TODO 
        # lprint 
        pass
    return res, cor 

def all(X,Y,x,y,args):
    return np.full(len(y),1) ,  np.full(len(y),1)


def VarianceInflation(X,Y,x,y,args):
    d=[]
    VIFALL = np.array([variance_inflation_factor(X, i) for i in range(X.shape[1])])
    for rep in Range(X):
        VIF = np.array([variance_inflation_factor(X, i) for i in range(X.shape[1])])
        print(f" {VIF}")
        if max(VIF)> 5:
            e = np.nanargmax(VIF)
            X[:,e] = 0
            d.append(e)
        else:
            break
    else:
        print(f"  something went wrong")

    #numpy.delete(arr, obj, axis=None) TODO remove one by one
    # 5 is the recommended cutoff, do i need to recalculate after removing each? molla
    # return negative VIF because when selecting in dataloader negative values are bad
    #print(f"  {VIF}")
    r = np.full(X.shape[1], True)
    r[d] = False
    so.lprint(r)
    return  r,-VIFALL


def main():
    args = opts.parse(featdoc)
    XYxy = datautil.getfold()
    np.savez_compressed(args.out, *eval(args.method)(*XYxy, args) ) 


if __name__ == "__main__":
    main()

