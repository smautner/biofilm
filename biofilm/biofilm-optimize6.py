import dirtyopts
import json
import biofilm.util.data as datautil
import numpy as np
import structout as so
from sklearn.metrics import  f1_score
import pprint
jdumpfile = lambda thing, filename:  open(filename,'w').write(json.dumps(thing))
from autosklearn.experimental.askl2 import AutoSklearn2Classifier as ASK2
from autosklearn.classification import AutoSklearnClassifier as ASK1
import autosklearn.metrics
import re 

optidoc='''
--method str any  'extra_trees', 'passive_aggressive', 'random_forest', 'sgd', 'gradient_boosting', 'mlp'
--out str jsongoeshere
--n_jobs int 1
--time int 3600
'''

def optimize(X,Y,x,y, args):
    if args.method == 'any':
        estis =  ['extra_trees', 'passive_aggressive', 'random_forest', 'sgd', 'gradient_boosting', 'mlp']
    else:
        estis = [args.method]
    estim = ASK1(
            include_estimators = estis,
            include_preprocessors = ["no_preprocessing"],
            n_jobs = args.n_jobs,
            ensemble_size = 1, 
            time_left_for_this_task = args.time,
            metric = autosklearn.metrics.f1,
            )
    estim.fit(X,Y)
    #import code
    #code.interact(local=dict(globals(), **locals()))
    # there is only 1 model in the end -> 0, we dont care about its weight -> 1 (this is the model) 
    #print(f" asdasdasd{estim.get_models_with_weights()}")
    estimator = estim.get_models_with_weights()[0][1] 
    return estimator # how do i get the model?




def get_params(ask): 
    a  =str(ask)
    classifier = re.findall("'classifier:__choice__': '(\w+)'",a)[0]
    args = re.findall(f"(classifier:{classifier}:[^,]+,)",a)
    return args
    


def main():

    args = dirtyopts.parse(optidoc)
    data, fea, ins = datautil.getfold()
    estim = optimize(*data,args)

    pred  = estim.predict(data[2])
    proba = estim.predict_proba(data[2])[:,1]
    score = f1_score(data[3],pred)



    #####
    # CSV: instance, reallabel, prediction, proba
    #######
    with open(args.out+".csv", "w") as f:
        things = zip(ins,data[3],pred,proba)
        things = [ f"{a}, {b}, {c}, {d}"  for a,b,c,d in things  ]
        f.write('\n'.join( things ) )
        f.write('\n')
    

    ###########
    # PRINT OUT
    #######
    print(f"{score=}") 
    pprint.pprint(get_params(estim))


    ##########
    # MODEL PARAMS
    ##########
    d={}
    d['score'] = score
    d['modelparams'] = get_params(estim)
    jdumpfile(d,args.out+'.model')
    



if __name__ == "__main__":
    main()

