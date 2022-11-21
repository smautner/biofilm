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


fnames = ['paris_human_RRI_',
 'full_',
  'full_human_',
   'paris_human_RBPs_',
    'splash_human_RRI_',
     'paris_splash_human_RRI_',
      'paris_mouse_RRI_']


'''
feature inspector is still bein built
'''
if what == 'inspectft':
    # in biofilm-features ftclust aufrufen um da wieder einzusteigen..
    python biofilm/biofilm-features.py --infile examples/1923MONOG --subsample 10000\
         --loader examples/cherriload.py --method agglocorr


'''
1. load data
we can do it in a fancy way, by providing a python file that has a
read(path) function. as demonstrated in examples/cherriload.py
'''
if what == 'runopti':
    loaddata += '--foldselect {1}'.split()
    parallel -j 5 --joblog opti.log $(which python) biofilm/optimize6.py  @(loaddata)\
        --out @(folder+'/{1}.optimized') --n_jobs 6 --time 36000 ::: $(seq 0 4)

if what == 'runopti2':
    loaddata += '--folds 0'.split()
    python biofilm/optimize6.py  @(loaddata)\
        --out @(folder+'/optimized') --preprocess True --n_jobs 30 --time 180 --subsample 1000

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
        parallel -j 5 --joblog delme.log $(which python) biofilm/biofilm-cv.py\
            @(loaddata) --model '{2}' --out '{2}_{1}.last'\
            ::: $(seq 0 4) ::: $(ls @(folder)/*optimized.model)

if what == 'rerunCV2':
        # rum models
        loaddata += '--foldselect {1}'.split()
        parallel -j 5 --joblog delme.log $(which python) biofilm/biofilm-cv.py\
            @(loaddata) --model @(folder)/optimized.model --out @(who)'{1}.lastcv2'\
            ::: $(seq 0 4)


if what == "crossspec":
    # after runopti2 and reruncv2 we have all the models we need
    #  .lastcv2 files have the CV data... so we need the rest
    for model in ["NOGHUMAN2","NOG","NOGMOUSE"]:
        if model != who:
            python biofilm/util/out.py --folds 0 @(loaddata)\
            --model @(model)/optimized.model --out @(f'CROSS_{model}_{who}')




if what == 'trueplot':
    import matplotlib
    matplotlib.use('module://matplotlib-sixel')
    import matplotlib.pyplot as plt
    import dill
    import pprint
    loadfile = lambda filename: dill.load(open(filename, 'rb'))
    for i in range(0,5):
        print(f"########### {i} ############")
        filez =  glob(f'{folder}/{i}*.last.csv')
        python biofilm/biofilm-out.py --infiles @(filez)
        d = loadfile(filez[0].split("_")[0]) # will load the model file
        print("self-score: ",d['score'])
        print("PARAMETERS")
        pprint.pprint(d['params'])
        if 'scorehistory' in d:
            plt.plot(d['scorehistory'])
            plt.show(); plt.close()


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


