from sklearn.neural_network import MLPClassifier as MLP
import dirtyopts
import dirtyopts as opts
import json
import biofilm.util.data as datautil
import numpy as np
import structout as so
from sklearn.metrics import  f1_score
import pprint
optidoc='''
--out str jsongoeshere
--n_jobs int 1
--loadmodel str ''
'''

jdumpfile = lambda thing, filename:  open(filename,'w').write(json.dumps(thing))
jloadfile = lambda filename:  json.loads(open(filename,'r').read())


def optimize(X,Y,x,y, args):
    
    if loadmodel ==  '':
        estim = MLP( alpha = 0.00002,
                #early_stopping = True, 
                epsilon = 1e-8, 
                hidden_layer_sizes =(60,60,60), 
                learning_rate_init = 0.0005,
                n_iter_no_change = 32,
                )
    else:
        estim= MLP()
        estim.set_params(jloadfile(args.loadmodel)['modelparams'])


    estim.fit(X,Y)
    oreo = estim.predict(x) 
    score = f1_score(y,oreo)
    proba =  estim.predict_proba(x)[:,1]

    return y,oreo, proba, skestim, score





def main():
    args = dirtyopts.parse(optidoc)
    data, fea, ins = datautil.getfold()
    y, pred, proba, skestim, score  = optimize(*data,args)

    
    #####
    # CSV: instance, reallabel, proba
    #######
    with open(args.out+".csv") as f:
        f.write('\n'.join([', ',join(zip( ins,y,pred,proba ))]) )
        f.write('\n')
    
    ##########
    # MODEL PARAMS
    ##########
    d={}
    d['score'] = score
    d['modelparams'] = estim.get_params()
    jdumpfile(d,args.out+'.model')
    


    ###########
    # PRINT OUT
    #######
    print(f"{score=}") 
    pprint.pprint(estim.get_params)

if __name__ == "__main__":
    main()



