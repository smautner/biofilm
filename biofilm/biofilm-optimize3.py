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


def bayes_search_CV_init(self, estimator, search_spaces, optimizer_kwargs=None,
                         n_iter=50, scoring=None, fit_params=None, n_jobs=1,
                         n_points=1, iid=True, refit=True, cv=None, verbose=0,
                         pre_dispatch='2*n_jobs', random_state=None,
                         error_score='raise', return_train_score=False):

        self.search_spaces = search_spaces
        self.n_iter = n_iter
        self.n_points = n_points
        self.random_state = random_state
        self.optimizer_kwargs = optimizer_kwargs
        self._check_search_space(self.search_spaces)
        self.fit_params = fit_params

        super(BayesSearchCV, self).__init__(
             estimator=estimator, scoring=scoring,
             n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
             pre_dispatch=pre_dispatch, error_score=error_score,
             return_train_score=return_train_score)
        
BayesSearchCV.__init__ = bayes_search_CV_init


def optimize(X,Y,x,y, args):
    clf, params = ss.classifiers[args.method]
    # Randomized search on hyper parameters
    #https://scikit-optimize.github.io/stable/auto_examples/sklearn-gridsearchcv-replacement.html
    searcher = BayesSearchCV(clf,params,n_iter=30,cv=3)
    searcher.fit(X, Y)
    #print (searcher.__dict__)
    print(f1_score(y, searcher.predict(x)))

if __name__ == "__main__":
    args = opts.parse(optidoc)
    data = datautil.getfolds()
    name = 'lol'
    np.savez_compressed(
            args.out, [optimize(X,Y,x,y,args) for X,Y,x,y in data])




