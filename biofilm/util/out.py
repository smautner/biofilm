
from sklearn.metrics import f1_score
import biofilm.util.data as datautil
import pprint
from biofilm import util



import re
def get_params(ask):
    '''
    this is supposed to work, however
    '''
    a  =str(ask)
    classifier = re.findall("'classifier:__choice__': '(\w+)'",a)[0]
    args = re.findall(f"(classifier:{classifier}:[^,]+,)",a)
    return args

def get_params2(ask):
    '''
    this should actually find the sklearn classifier and get the paramns
    '''
    args=str(ask.steps[-1][1].choice.__dict__)

    return args

def report(estim, outputname, quiet=False, predict_train=False,additionaloutput={}):
    '''
    dumps the csv file
    dumps the model
    '''
    data, fea, ins = datautil.getfold()
    dataargs = datautil.getargs()
    params = get_params2(estim)
    if predict_train:
        X = data[0]
        y = data[1]
    else:
        X = data[2]
        y= data[3]

    pred  = estim.predict(X)
    proba = estim.predict_proba(X)[:,1]
    score = f1_score(y,pred)

    #####
    # CSV: instance, reallabel, prediction, proba
    #######
    with open(outputname+".csv", "w") as f:
        things = zip(ins,y,pred,proba)
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
    d = {}
    d['score'] = score
    d['params'] = params
    d['estimator'] = estim
    d.update(additionaloutput)
    util.dumpfile(d, outputname+'.model')


optidoc='''
--out str outname
--model str inputmodel
--predict_train bool False
'''
import dirtyopts
args = dirtyopts.parse(optidoc)

if __name__ == "__main__":
    mod = util.loadfile(args.model)
    report(mod, args.out,predict_train = args.predict_train )

