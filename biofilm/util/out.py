
from sklearn.metrics import f1_score
import biofilm.util.data as datautil
import pprint
import dirtyopts
from ubergauss import tools



import re
optidoc='''
--out str outname
--model str inputmodel
--predict_train bool False
'''



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
    #ask = ask.get_models_with_weights()[0][1]
    if 'steps' not in ask.__dict__:
        print('AUTOSKLEARN DID NOT FIND A MODEL')
        exit()
    args=str(ask.steps[-1][1].choice.__dict__)
    try:
        # not sure if a choice is required
        args+='\n\n\n\n '+ str(ask.steps[-2][1].choice.__dict__)
    except:
        args+='\n failed to get preprocessing info'
    return args

def report(model, outputname, quiet=False, predict_train=False,additionaloutput={}):
    '''
    dumps the csv file
    dumps the model
    '''
    pipeline = model.get_models_with_weights()[0][1]


    ####
    # run on test set
    #########
    data, fea, ins = datautil.getfold()
    dataargs = datautil.getargs()
    params = get_params2(pipeline)

    if predict_train or data[2].shape[0] ==0:
        print("INFO: there is no test set provided so we eval on train. this may be unintendet.")
        ins = ins[f'train']
        X = data[0]
        y = data[1]
    else:
        ins = ins[f'test']
        X = data[2]
        y= data[3]

    pred  = pipeline.predict(X)
    proba = pipeline.predict_proba(X)[:,1]
    score = f1_score(y,pred)

    #####
    # CSV: instance, reallabel, prediction, proba
    #######
    with open(outputname+".csv", "w") as f:
        things = zip(ins,y,pred,proba)
        things = [ f"{a}, {b}, {c}, {d}, {dataargs.randinit}"  for a,b,c,d in things  ]
        things = ['instance_id,true_label,predicted_label,instance_score,rand_init'] + things
        f.write('\n'.join( things ) )
        f.write('\n')
        print("\n########## CSV WRITTEN ##########\n")

    ###########
    # PRINT OUT
    #######
    if not quiet:
        print(f"TEST {score=}")
        pprint.pprint(params)

    ##########
    # MODEL PARAMS
    ##########
    d = {}
    d['score'] = score
    d['params'] = params
    d['estimator'] = model

    d.update(additionaloutput)
    tools.dumpfile(d, outputname+'.model')
    print("\n########## MODEL WRITTEN ##########\n")



if __name__ == "__main__":
    args = dirtyopts.parse(optidoc)
    mod = tools.loadfile(args.model)
    if type(mod)==dict:
        report(mod['estimator'], args.out,predict_train = args.predict_train )
    else: # TODO  we should never arrive here, but we do, this should be solved at some opint
        report(mod, args.out,predict_train = args.predict_train )
