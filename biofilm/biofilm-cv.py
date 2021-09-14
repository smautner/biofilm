import dirtyopts
import biofilm.util.out as out
import dill
from biofilm.util import data as datautil
optidoc='''
--out str modeldilldump
--model str inputmodel
'''


loadfile = lambda filename: dill.load(open(filename, 'rb'))

def fit(X,Y,x,y, args):
    print(f"{ args=}")
    model = loadfile(args.model)['estimator']
    model.fit(X,Y)
    return model

def main():
    args = dirtyopts.parse(optidoc)
    data, fea, ins = datautil.getfold()
    estim = fit(*data,args)
    out.report(estim, args.out)

if __name__ == "__main__":
    main()

