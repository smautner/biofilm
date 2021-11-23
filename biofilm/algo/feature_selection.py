from lmz import Map,Zip,Filter,Grouper,Range,Transpose
import dirtyopts as opts
import matplotlib
matplotlib.use('module://matplotlib-sixel')
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import numpy as np
from sklearn.linear_model import SGDClassifier as sgd, LassoCV, LassoLarsCV, LinearRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
import structout as so
import ubergauss
from scipy.stats import spearmanr
import biofilm.util.draw as draw
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from biofilm import util
from biofilm import algo
from biofilm.algo import feature_inspection
from sklearn.ensemble import RandomForestClassifier
from biofilm import algo, util




def relaxedlasso(X,Y,x,y,args):
    print("RELAXED LASSO NOT IMPLEMENTD ") # TODO


def _forest(X, Y, **kwargs):
    model = RandomForestClassifier(**kwargs).fit(X,Y)
    quality = model.feature_importances_
    return  quality, model
def forest(X,Y,x,y,args):
    scores, model = algo.feature_selection._forest(class_weight='balanced', random_state = 0)
    res =  autothresh(scores)[0]
    # TODO I THINK I SHOULD RERAIN>>> or just use the default linear model...
    trainscore   = f1_score(Y, model.predict(X))
    testscore   = f1_score(y, model.predict(x))
    print(f" {trainscore=:.2f} {testscore=:.2f}")
    return res, scores


def lasso(X,Y,x,y,args):
    model = LassoCV(n_alphas = 100,n_jobs = args.n_jobs).fit(X,Y)
    quality = abs(model.coef_)
    res =  quality > 0.0001

    testscore, cutoff = max([(f1_score(Y,model.predict(X) > t),t) for t in np.arange(.3,.7,.001)])
    print ('score: %.2f alpha: %.4f  features: %d/%d  ERRORPATH: ' %
            ( f1_score(y,model.predict(x)>cutoff), model.alpha_, (model.coef_>0.0001).sum(), len(model.coef_)), end= '')

    so.lprint(model.mse_path_.mean(axis = 0))

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
    return res, quality


def lassolars(X,Y,x,y,args):
    model = LassoLarsCV(n_jobs = args.n_jobs).fit(X,Y)
    quality = abs(model.coef_)
    res =  quality > 0.0001


    testscore, cutoff = max([(f1_score(Y,model.predict(X) > t),t) for t in np.arange(.3,.7,.001)])
    print ('score: %.2f alpha: %.4f  features: %d/%d errorpath: ' %
            ( f1_score(y,model.predict(x)>cutoff), model.alpha_, (model.coef_>0.0001).sum(), len(model.coef_)), end ='')
    so.lprint(model.mse_path_.mean(axis = 0))


    return res, quality


def svm(X,Y,x,y,args, quiet = False):
    clf = LinearSVC(class_weight = 'balanced', max_iter=1000)
    param_dist = {"C": np.logspace(*args.svmparamrange[:2], int(args.svmparamrange[2])) ,
            'penalty':[args.penalty],'dual':[False]}


    search = GridSearchCV(clf,param_dist, n_jobs=args.n_jobs, scoring='f1', cv = 3).fit(X,Y)
    model = search.best_estimator_
    err = search.cv_results_["mean_test_score"]
    if not quiet:
        print ("numft %d/%d  C %.3f score %.3f scorepath: " %
                ((abs(model.coef_)>0.0001 ).sum(),
                    len(model.coef_.ravel()),
                    model.C,f1_score(y,model.predict(x))), end='')

        #so.lprint(err, length = 25, minmax = True)
        so.lprint(err)

    quality = abs(model.coef_)
    res = ( quality > 0.0001).ravel()#squeeze()
    return res, quality


def autothresh(arr, cov = 'tied'):
    arr=np.abs(arr)
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
    var.sort()
    so.lprint(var, length = 50)

    if args.plot:
        plt.plot(var)
        plt.show()

    return res, var

from ubergauss import tools

def corr(X,Y,x,y,args):
    X = tools.zehidense(X)

    if type(X) == sparse.csr_matrix:
        #X2 = sparse.csc_matrix(X)
        X2 = X.todense().T
        def spr(x):
            res = abs(spearmanr(x.A1,Y)[0])
            return 0 if np.isnan(res) else res
        cor = np.array([spr(x) for x in X2])
    else:
        # todo ...THIS DEFINITELY FAILS BUT I DONT WANT TO FIX IT  NOW
        cor = abs([spearmanr(X2[:,column].todense().A1,Y)[0] for column in range(X.shape[1])])

    res, cut= autothresh(cor)
    print(f"cor  features: {sum(res)}/{len(res)} ",end ='')
    so.lprint(cor)

    if args.plot:
        c2=np.sort(cor)
        plt.title(f"cut: {cut}")
        plt.plot(c2)
        plt.show()

    return res, cor


def all(X,Y,x,y,args):
    res = np.full( X.shape[1], True)
    return res,res


def agglocore(X,Y,x,y,args):

    # fit agglo
    numft = X.shape[1]
    clf = AgglomerativeClustering(n_clusters = min(100,numft) ,compute_distances=True)
    X_data = np.transpose(util.zehidense(X))
    clf.fit(X_data)

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
    X=X.todense()
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


def agglosvm(X,Y,x,y,args):
    res,_ = agglocore(X,Y,x,y,args)
    res = np.array(res) == True
    X2 = X[:,res]
    x2 = x[:,res]
    caccept, _ = svm(X2,Y,x2, y, args, quiet = True)
    res[res == 1] = caccept
    print(f"aglo+ features: {sum(res)}/{len(res)} ",end ='')
    return res, np.full(X.shape[1],1)