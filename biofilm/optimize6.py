import numpy as np
import dirtyopts
import structout as so
from sklearn.metrics import  f1_score
import pprint
from biofilm import util
from autosklearn.experimental.askl2 import AutoSklearn2Classifier as ASK2
from autosklearn.classification import AutoSklearnClassifier as ASK1
import autosklearn.metrics

optidoc='''
--methods str+ any  'extra_trees', 'passive_aggressive', 'random_forest', 'sgd', 'gradient_boosting', 'mlp'
--out str jsongoeshere
--n_jobs int 1
--time int 3600
--memoryMBthread int 8000
--randinit int 1337  # should be the same as the one in data.py
--preprocess bool False
--tmp_folder str
#--metric str f1 assert f1 auc   TODO
'''

'''
this gives all the calssifiers:
import autosklearn.pipeline.components.classification as cls
cls._classifiers.keys()
'adaboost', 'bernoulli_nb', 'decision_tree', 'extra_trees', 'gaussian_nb', 'gradient_boosting', 'k_nearest_neighbors', 'lda', 'liblinear_svc', 'libsvm_svc', 'mlp', 'multinomial_nb', 'passive_aggressive', 'qda', 'random_forest', 'sgd'
'''

def optimize(X,Y,x,y, args):

    if args.methods[0] == 'any':
        estis =  None #['extra_trees', 'passive_aggressive', 'random_forest', 'sgd', 'gradient_boosting', 'mlp']
    else:
        estis = args.methods


    #include_estimators = estis,
    #include_preprocessors = ["no_preprocessing"] if not args.preprocess else None,

    include = {'classifier': estis} if estis else {}
    print("ALLGOOD:", include)
    if not args.preprocess:
            include['feature_preprocessor'] =  ["no_preprocessing"]
            #include['data_preprocessor']     =  ['NoPreprocessing']

    estim = ASK1(
            n_jobs = args.n_jobs,
            ensemble_size = 1,
            include = include,
            memory_limit = args.memoryMBthread,
            time_left_for_this_task = args.time,
            metric = autosklearn.metrics.f1,
            max_models_on_disc = 1,
            tmp_folder = args.tmp_folder or None,
            initial_configurations_via_metalearning=0 # autosklearn thros warnings otherwise
            )

    print('OPTIMIZATION DATATYPE:',type(x))
    estim.fit(x,y)
    #IMPORT CODE
    #code.interact(local=dict(globals(), **locals()))
    # there is only 1 model in the end -> 0, we dont care about its weight -> 1 (this is the model)
    #print(f" asdasdasd{estim.get_models_with_weights()}")
    # models with weights is a list of model in the ensemble: [(weight_1, model_1), …, (weight_n, model_n)]
    #pipeline = estim.get_models_with_weights()[0][1]
    #print('estim',estim.get_params(deep=True))
    #print('estimator:',estimator.get_params(deep=True))

    return estim




def main():
    args = dirtyopts.parse(optidoc)
    data, fea, ins = util.getfold()
    model = optimize(*data,args)
    scorehistory =  np.nan_to_num(\
            model.performance_over_time_['single_best_optimization_score'].to_numpy(),nan=0.0)
    util.report(model, args.out, additionaloutput=\
            {'scorehistory': scorehistory , 'performancelog': model.performance_over_time_})
    so.lprint(scorehistory)
    if data[0].shape[1] < 100:
        print("SELECTED FEATURES: ", end='')
        pipeline = model.get_models_with_weights()[0][1]
        try:
            if type(pipeline.steps[2][1].choice.preprocessor) == str:
                print("all")
            else:
                print('\n',pipeline.steps[2][1].choice.preprocessor.get_support())
        except:
            pass


if __name__ == "__main__":
    main()

