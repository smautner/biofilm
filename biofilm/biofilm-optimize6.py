import dirtyopts as opts
import json
import biofilm.util.data as datautil
import numpy as np
import structout as so
from sklearn.metrics import  f1_score
import pprint

from autosklearn.experimental.askl2 import AutoSklearn2Classifier as ASK2
from autosklearn.classification import AutoSklearnClassifier as ASK1
import autosklearn.metrics
optidoc='''
--method str any  'extra_trees', 'passive_aggressive', 'random_forest', 'sgd', 'gradient_boosting', 'mlp'
--out str jsongoeshere
--n_jobs int 1
--debug bool False
'''

from hpsklearn.components import *
#pip install git+https://github.com/hyperopt/hyperopt-sklearn 
def optimize(X,Y,x,y, args):
    #estim = autosklearn.classification.AutoSklearnClassifier()
    if args.method == 'any':
        estis =  ['extra_trees', 'passive_aggressive', 'random_forest', 'sgd', 'gradient_boosting', 'mlp']
    else:
        estis = [args.method]

    estim = ASK1(
            include_estimators = estis,
            include_preprocessors = ["no_preprocessing"],
            n_jobs = args.n_jobs,
            ensemble_size = 1, 
            time_left_for_this_task = 600 if args.debug else 21600,
            metric = autosklearn.metrics.f1,
            )
    estim.fit(X,Y)
    score = f1_score(y,estim.predict(x) )
    parm = str(estim.show_models())
    res = {'score': score, "param":parm}
    pprint.pprint(res)
    return res

jdumpfile = lambda thing, filename:  open(filename,'w').write(json.dumps(thing))

def main():
    args = opts.parse(optidoc)
    data = datautil.getfold()
    res = optimize(*data,args)
    jdumpfile(res,args.out)

if __name__ == "__main__":
    main()



