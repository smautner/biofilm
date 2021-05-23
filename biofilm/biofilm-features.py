import dirtyopts as opts
import util.data as datautil
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
import util.draw as draw
import matplotlib.pyplot as plt


featdoc='''
--method str lasso  or svm or all or corr or variance
--out str numpycompressdumpgoeshere
--plot bool False

--svmparamrange float+ 0.01 0.15 0.001
'''

def lasso(X,Y,x,y,args):
    model = LassoCV(n_alphas = 100).fit(X,Y)
    if args.plot:
        testscore, cutoff = max([(f1_score(Y,model.predict(X) > t),t) for t in np.arange(.3,.7,.001)])
        print ('score: %.2f alpha: %.4f  features: %d  ERRORPATH, FEATCOUNT' % ( f1_score(y,model.predict(x)>cutoff), model.alpha_, (model.coef_>0.0001).sum()))
        draw.lasso(model,X,Y)
    res =  model.coef_ > 0.0001
    so.lprint(res.astype(np.int64))
    return res


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
    res = ( model.coef_ > 0.0001).squeeze()
    so.lprint(res.astype(np.int64))
    return res

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
    return res

def corr(X,Y,x,y,args):
    cor = np.array([spearmanr(X[:,column],Y)[0] for column in range(X.shape[1])])
    res = (autothresh(cor))
    so.lprint(res.astype(np.int64))
    return res

def all(X,y,args):
    return np.full(len(y),1)



if __name__ == "__main__":
    args = opts.parse(featdoc)
    data = datautil.getfolds()
    np.savez_compressed(
            args.out, [eval(args.method)(X,Y,x,y,args) for X,Y,x,y in data])

