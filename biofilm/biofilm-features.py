import dirtyopts as opts
import matplotlib
matplotlib.use('module://matplotlib-sixel')
import matplotlib.pyplot as plt


from lmz import *

import biofilm.util.data as datautil
import numpy as np
from sklearn.linear_model import SGDClassifier as sgd, LassoCV, LassoLarsCV, LinearRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV
import structout as so
import random
import warnings
import ubergauss
from scipy.stats import spearmanr
import biofilm.util.draw as draw
from sklearn.linear_model import LogisticRegressionCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron, SGDClassifier
featdoc=''' 
# options for feature selection:
--method str lasso  svm all corr variance logistic relaxedlasso  VarianceInflation aggloclust
--out str numpycompressdumpgoeshere
--plot bool False
--n_jobs int 1

--svmparamrange float+ 0.01 .3 50
--penalty str l1
--varthresh float 1
'''
def relaxedlasso(X,Y,x,y,args):
    print("RELAXED LASSO NOT IMPLEMENTD ") # TODO 

def lasso(X,Y,x,y,args):
    model = LassoCV(n_alphas = 100,n_jobs = args.n_jobs).fit(X,Y)
    quality = abs(model.coef_)
    res =  quality > 0.0001

    testscore, cutoff = max([(f1_score(Y,model.predict(X) > t),t) for t in np.arange(.3,.7,.001)])
    print ('score: %.2f alpha: %.4f  features: %d/%d  ERRORPATH: ' % 
            ( f1_score(y,model.predict(x)>cutoff), model.alpha_, (model.coef_>0.0001).sum(), len(model.coef_)), end= '')

    so.lprint(model.mse_path_.mean(axis = 0))
    #so.lprint(res.astype(np.int64))

    return res, quality

def logistic(X,Y,x,y,args):
    model = LogisticRegressionCV(Cs=10, 
                        penalty = args.penalty, 
                        max_iter = 300, 
                        solver ='liblinear',
                        n_jobs = args.n_jobs).fit(X,Y)
    quality = abs(model.coef_.ravel())
    res =  quality > 0.0001

    print(f"  score:{f1_score(y,model.predict(x))}  feaures: {sum(res)}/{len(res)} ")
    #so.lprint(res.astype(np.int64))
    return res, quality

def lassolars(X,Y,x,y,args):
    model = LassoLarsCV(n_jobs = args.n_jobs).fit(X,Y)
    quality = abs(model.coef_)
    res =  quality > 0.0001


    testscore, cutoff = max([(f1_score(Y,model.predict(X) > t),t) for t in np.arange(.3,.7,.001)])
    print ('score: %.2f alpha: %.4f  features: %d/%d errorpath: ' %
            ( f1_score(y,model.predict(x)>cutoff), model.alpha_, (model.coef_>0.0001).sum(), len(model.coef_)), end ='')
    so.lprint(model.mse_path_.mean(axis = 0))


    #so.lprint(res.astype(np.int64))
    return res, quality

def svm(X,Y,x,y,args): 
    clf = LinearSVC(class_weight = 'balanced', max_iter=500)
    param_dist = {"C": np.linspace(*args.svmparamrange[:2], int(args.svmparamrange[2])) ,
            'penalty':[args.penalty],'dual':[False]}


    search = GridSearchCV(clf,param_dist, n_jobs=args.n_jobs, scoring='f1', cv = 3).fit(X,Y)
    model = search.best_estimator_

    print ("numft %d/%d  C %.3f score %.3f scorepath: " % 
            ((abs(model.coef_)>0.0001 ).sum(),len(model.coef_.ravel()),model.C,f1_score(y,model.predict(x))), end='')

    err = search.cv_results_["mean_test_score"]
    so.lprint(err, length = 25)

    quality = abs(model.coef_)
    res = ( quality > 0.0001).squeeze()
    #so.lprint(res.astype(np.int64))
    return res, quality

def autothresh(arr, cov = 'tied'):
    arr=abs(arr)
    cpy = np.array(arr)
    cpy.sort()
    rr = ubergauss.between_gaussians(cpy, covariance_type = cov)
    return arr >= cpy[rr] , cpy[rr]

def variance(X,Y,x,y,args):
    var = np.var(X, axis = 0)
    if args.varthresh <= 0: 
        res = (autothresh(var)[0])
    else:
        res = var > args.varthresh

    print(f"var  features: {sum(res)}/{len(res)} ",end =' ')
    #so.lprint(res.astype(np.int64))
    var.sort()
    so.lprint(var, length = 50)

    if args.plot:
        plt.plot(var)
        plt.show()

    #print(res.astype(np.int64))
    return res, var

def corr(X,Y,x,y,args):
    cor = abs(np.array([spearmanr(X[:,column],Y)[0] for column in range(X.shape[1])]))
    res, cut= autothresh(cor)
    print(f"cor  features: {sum(res)}/{len(res)} ",end ='')
    cor.sort() 
    so.lprint(cor, length = 50)    

    if args.plot:
        plt.title(f"cut: {cut}")
        plt.plot(cor)
        plt.show()

    return res, cor 

def all(X,Y,x,y,args):
    return np.full(X.shape[1],1) ,  np.full(X.shape[1],1)



def agglocore(X,Y,x,y,args):
    clf = AgglomerativeClustering(n_clusters = 100,compute_distances=True) 
    X_data = np.transpose(X)
    clf.fit(X_data)
    
    numft = X.shape[1]
    dists = np.array([a for a,(b,c) in zip(clf.distances_, clf.children_) if b < c < numft])
    _, mydist = autothresh(dists,'tied')

    if args.plot:
        plt.title(f"cut: {mydist}")
        dists.sort()
        plt.plot(dists)
        plt.show()

    clf = AgglomerativeClustering(distance_threshold  =  mydist)
    clf.n_clusters = None
    clf.fit(X_data)

    labels = clf.labels_
    uni = np.unique(labels)
    fl = []
    for i in uni:
        clusterinstances = np.where(labels == i)[0]
        erm = [np.abs(spearmanr(Y, X[:,f])[0]) for f in clusterinstances]
        zzz = np.full(len(erm), 0 )
        zzz[np.argmax(erm)]  = 1
        fl.append(zzz) 
    res = np.hstack(fl)

    print(f"agloc features: {sum(res)}/{len(res)} ",end ='')
    return res, np.full(X.shape[1],1)


def agglocorr(X,Y,x,y,args):
    res,_ = agglocore(X,Y,x,y,args)
    cor = abs(np.array([spearmanr(X[:,column],Y)[0] for column in [i for i,e in enumerate(res) if e ]]))
    caccept, cut = autothresh(cor, cov = 'full')
    res[res == 1] = caccept
    print(f"aglo+ features: {sum(res)}/{len(res)} ",end ='')
    if args.plot:
        plt.close()
        plt.title(f"cut: {cut}")
        cor.sort()
        plt.plot(cor)
        plt.show()
    return res, np.full(X.shape[1],1)


def main():
    args = opts.parse(featdoc)
    XYxy = datautil.getfold()
    res  = eval(args.method)(*XYxy, args) 
    #import pprint;pprint.pprint(res)
    np.savez_compressed(args.out,*res)


if __name__ == "__main__":
    main()

