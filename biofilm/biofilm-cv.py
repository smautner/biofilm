import dirtyopts
import biofilm.util.out as out
import dill
from biofilm.util import data as datautil

optidoc='''
--out str modeldilldump
--model str inputmodel
'''


loadfile = lambda filename: dill.load(open(filename, 'rb'))
dumpfile  = lambda thing, fn: dill.dump(thing,open(fn,'wb'))


def fit(X,Y,x,y, args):
    model = loadfile(args.model)['estimator']
    model.fit(X,Y)
    return model

def main():
    args = dirtyopts.parse(optidoc)
    data, fea, ins = datautil.getfold()
    estim = fit(*data,args)

    if data[2].shape[0]==0:
        print('there is no test set so we just dump the model')
        dumpfile( estim ,args.out+'.model')
    else:
        out.report(estim, args.out)

if __name__ == "__main__":
    main()

