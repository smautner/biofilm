

import pprint
import json
import dill
jdumpfile = lambda thing, filename:  open(filename,'w').write(json.dumps(thing))
dumpfile  = lambda thing, fn: dill.dump(thing,open(fn,'wb'))
from sklearn.metrics import  f1_score
import biofilm.util.data as datautil

import re
def get_params(ask):
    a  =str(ask)
    classifier = re.findall("'classifier:__choice__': '(\w+)'",a)[0]
    args = re.findall(f"(classifier:{classifier}:[^,]+,)",a)
    return args

def report(estim, outputname, quiet = False):
    '''
    dumps the csv file
    dumps the model
    '''
    data, fea, ins = datautil.getfold()
    dataargs = datautil.getargs()
    params = get_params(estim)


    pred  = estim.predict(data[2])
    proba = estim.predict_proba(data[2])[:,1]
    score = f1_score(data[3],pred)

    #####
    # CSV: instance, reallabel, prediction, proba
    #######
    with open(outputname+".csv", "w") as f:
        things = zip(ins,data[3],pred,proba)
        things = [ f"{a}, {b}, {c}, {d}, {dataargs.randinit}"  for a,b,c,d in things  ]
        things = ['instance_id, true_label, predicted_label, instance_score, rand_init'] + things
        f.write('\n'.join( things ) )
        f.write('\n')


    ###########
    # PRINT OUT
    #######
    if not quiet:
        print(f"{score=}")
        pprint.pprint(params)

    ##########
    # MODEL PARAMS
    ##########
    d={}
    d['score'] = score
    d['params'] = params
    d['estimator'] = estim
    dumpfile(d,outputname+'.model')
