

import pprint
import json
jdumpfile = lambda thing, filename:  open(filename,'w').write(json.dumps(thing))
from sklearn.metrics import  f1_score
import biofilm.util.data as datautil


def report(estim, params, args, quiet = False):
    data, fea, ins = datautil.getfold()
    dataargs = datautil.getargs()

    pred  = estim.predict(data[2])
    proba = estim.predict_proba(data[2])[:,1]
    score = f1_score(data[3],pred)

    #####
    # CSV: instance, reallabel, prediction, proba
    #######
    with open(args.out+".csv", "w") as f:
        things = zip(ins,data[3],pred,proba)
        things = [ f"{a}, {b}, {c}, {d}, {dataargs.randinit}"  for a,b,c,d in things  ]
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
    d['modelparams'] = params
    jdumpfile(d,args.out+'.model')
