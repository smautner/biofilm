'''
1. load data
we can do it in a fancy way, by providing a python file that has a
read(path) function. as demonstrated in examples/cherriload.py
'''
loaddata = '--infile examples/2291HU --loader examples/cherriload.py '

'''
2. extract features
'''
import sys
what = sys.argv[1]
if what=='ftselect':
    parallel -j 32 --joblog feat.log python biofilm/biofilm-features.py @(loaddata)\
        --foldselect '{1}' --method forest --out 'bigcherry/{1}' ::: $(seq 0 4)

'''
3. run optimization
-> outputs crossvall results in .csv files and model params in .model
!!! carefull with the memory limit, process will die without terminating...
'''
loaddata += '--featurefile bigcherry/{1} --foldselect {1}'
if what == 'runopti':
    parallel -j 5 --joblog opti.log python biofilm/biofilm-optimize6.py  @(loaddata)\
        --out 'bigcherry/{1}.optimized' --n_jobs 6 --time 54000 ::: $(seq 0 4)

'''
4. plot performance (so far)
'''
if what == 'plot1':
    python biofilm/biofilm-out.py --infiles bigcherry/*.csv


'''
5. do crossval for all models
      use all 5 models to crossvalidate over all instances to compare them...
'''
if what == 'rerunCV':
    # rum models
    parallel -j 32 --joblog delme.log python biofilm/biofilm-cv.py  @(loaddata) --model '{2}'\
        --out '{2}_{1}.last' ::: $(seq 0 4) ::: $(ls bigcherry/*optimized.model)

import dill
loadfile = lambda filename: dill.load(open(filename, 'rb'))
if what == 'trueplot':
    import pprint
    import glob
    for i in range(0,4):
        search = f'bigcherry/{i}*last.csv'
        filez=glob.glob(search)
        if filez:
            print(filez)
            python biofilm/biofilm-out.py --infiles @(filez) --rawproba True
            d = loadfile(filez[0].split("_")[0])
            print("self-score: ",d['score'])
            print("PARAMETERS")
            pprint.pprint(d['params'])

if what == 'refit':
    python biofilm/biofilm-cv.py  --folds 0 --featurefile bigcherry/3 \
        --infile examples/2291HU --loader examples/cherriload.py \
        --model bigcherry/3.optimized.model --out 'UBERMODEL'

if what == 'mouseeval':
    python biofilm/util/out.py --model UBERMODEL.model --out MOUSEOUT\
        --infile examples/1923MO --loader examples/cherriload.py\
        --folds 0 --featurefile bigcherry/3 --predict_train True

if what ==  'mousedraw':
    python biofilm/biofilm-out.py --infiles "MOUSEOUT.csv"

