

import sys
from glob import glob
what =  sys.argv[1]


loaddata = '--infile examples/2291HUNOG --loader examples/cherriload.py '
folder = 'NOG'
if sys.argv[2] == 'mouse':
    loaddata = '--infile examples/1923MONOG --loader examples/cherriload.py '
    folder = 'NOGMOUSE'

if what == 'inspectft':
    python biofilm/biofilm-features.py --infile examples/2291HU --subsample 2000\
         --loader examples/cherriload.py --method agglocorr


'''
1. load data
we can do it in a fancy way, by providing a python file that has a
read(path) function. as demonstrated in examples/cherriload.py
'''


if what == 'runopti':
    parallel -j 5 --joblog opti.log python biofilm/biofilm-optimize6.py  @(loaddata)\
        --out @(folder)'/{1}.optimized' --n_jobs 6 --time 72000 ::: $(seq 0 4)




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
    parallel -j 32 --joblog delme.log python biofilm/biofilm-cv.py  @(loaddata) --model '{2}'\
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






if what == 'makehumanmodel':
    python biofilm/biofilm-cv.py  --folds 0\
        --infile examples/2291HUNOG --loader examples/cherriload.py \
        --model NOG/2.optimized.model --out 'UBERMODELNOG'

if what == 'makemousemodel':
    python biofilm/biofilm-cv.py  --folds 0\
        --infile examples/1923MONOG --loader examples/cherriload.py \
        --model NOGMOUSE/1.optimized.model --out 'NOGMOUSE'

if what == 'evalmouseonhuman':
    python biofilm/util/out.py --model UBERMODELNOG.model --out MOUSEOUTNOG\
        --infile examples/1923MONOG --loader examples/cherriload.py\
        --folds 0 --predict_train True
    python biofilm/biofilm-out.py --infiles "MOUSEOUTNOG.csv"


