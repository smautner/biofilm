import sys
from glob import glob
what =  sys.argv[1]
who =  sys.argv[2]

if who == 'all':
    xonsh cherryexampleNOG.xsh @(what) mouse
    xonsh cherryexampleNOG.xsh @(what) human
    xonsh cherryexampleNOG.xsh @(what) human2
    exit()

loaddata = '--infile examples/2291HUNOG --loader examples/cherriload.py '
folder = 'NOG'
if who == 'mouse':
    loaddata = '--infile examples/1923MONOG --loader examples/cherriload.py '
    folder = 'NOGMOUSE'

if who == 'human2':
    loaddata = '--infile examples/HUMANRBPNOG --loader examples/cherriload.py '
    folder = 'NOGHUMAN2'
loaddata = loaddata.split()



if what == 'inspectft':
    python biofilm/biofilm-features.py --infile examples/1923MONOG --subsample 10000\
         --loader examples/cherriload.py --method agglocorr


'''
1. load data
we can do it in a fancy way, by providing a python file that has a
read(path) function. as demonstrated in examples/cherriload.py
'''
if what == 'runopti':
    loaddata += '--foldselect {1}'.split()
    parallel -j 5 --joblog opti.log $(which python) biofilm/biofilm-optimize6.py  @(loaddata)\
        --out @(folder+'/{1}.optimized') --n_jobs 6 --time 36000 ::: $(seq 0 4)


'''
4. plot performance (so far)
'''
if what == 'plot1':
    python biofilm/biofilm-out.py --infiles  @(glob(f'{folder}/*.csv'))



'''
5. do crossval for all models
      use all 5 models to crossvalidate over all instances to compare them...
'''
if what == 'rerunCV':
        # rum models
        loaddata += '--foldselect {1}'.split()
        parallel -j 32 --joblog delme.log $(which python) biofilm/biofilm-cv.py  @(loaddata) --model '{2}'\
            --out '{2}_{1}.last' ::: $(seq 0 4) ::: $(ls @(folder)/*optimized.model)

if what == 'trueplot':
    import dill
    import pprint
    loadfile = lambda filename: dill.load(open(filename, 'rb'))
    for i in range(0,5):
        filez =  glob(f'{folder}/{i}*.last.csv')
        python biofilm/biofilm-out.py --infiles @(filez)
        d = loadfile(filez[0].split("_")[0])
        print("self-score: ",d['score'])
        print("PARAMETERS")
        pprint.pprint(d['params'])


if what == 'finalmodel':
    model= {
        'human': folder+'/0.optimized.model',
        'human2': folder+'/2.optimized.model',
        'mouse':  folder+'/2.optimized.model'
    }
    python biofilm/biofilm-cv.py --folds 0 @(loaddata)\
        --model @(model[who]) --out @('UBERMODEL_'+who)


if what == 'evalmouseonhuman':
    python biofilm/util/out.py --model UBERMODELNOG.model --out MOUSEOUTNOG\
        --infile examples/1923MONOG --loader examples/cherriload.py\
        --folds 0 --predict_train True
    python biofilm/biofilm-out.py --infiles "MOUSEOUTNOG.csv"


