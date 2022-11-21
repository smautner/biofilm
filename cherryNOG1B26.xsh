import sys
from glob import glob
what =  sys.argv[1]
dataset =  sys.argv[2]

'''
script to train and evaluate the rnarnainteration data

use like this:
    xonsh cherryNOG1B26.xsh makedata all
    xonsh cherryNOG1B26.xsh optimize all
    xonsh cherryNOG1B26.xsh runcv all
    xonsh cherryNOG1B26.xsh refit all
    xonsh cherryNOG1B26.xsh crossmodel all
'''

fnames = ['paris_human_RRI_',
 'full_',
  'full_human_',
   'paris_human_RBPs_',
    'splash_human_RRI_',
     'paris_splash_human_RRI_',
      'paris_mouse_RRI_']


if dataset == 'all':
    for target in fnames:
        xonsh cherryNOG1B26.xsh @(what) @(target)
    exit()



# ok lets make some DATA
if what == 'makedata':
    print(f'makedata {dataset}')
    mkdir -p NOG/data/
    # TODO change imoprt blabla to biofilm.examples.cherriload
    import examples.cherriload  as cl
    p = "/home/ubuntu/data/cherry/"+dataset
    d1 = p+'neg.csv'
    d2 = p+"pos.csv"
    cl.convert(d1,d2,f'NOG/data/{dataset}', graphfeatures=False)

loaddata = f'--infile NOG/data/{dataset} --loader examples/cherriload.py '.split()


'''
2. LETZ OPTYIMIZE
'''
if what == 'optimize':
    mkdir -p NOG/optimized
    loaddata += '--folds 0 --subsample 10000'.split()
    # TODO python -m biofilm.optimize6 should work
    python biofilm/optimize6.py  @(loaddata)\
        --out @(f'NOG/optimized/{dataset}') --preprocess True --n_jobs 30 --time 28800





'''
retrain and cv to get all the csv files
'''

if what == 'runcv':
    # rum models
    loaddata += '--foldselect {1}'.split()
    # TODO python -m biofilm.biofilm-cv
    parallel -j 5 --joblog delme.log $(which python) biofilm/biofilm-cv.py @(loaddata)\
        --model @('NOG/optimized/%s.model' % dataset)\
        --out @('NOG/crossval/%s{1}.cv' % dataset)\
        ::: $(seq 0 4)

if what == "refit":
    mkdir -p NOG/refit
    python -m biofilm.biofilm-cv --folds 0 @(loaddata)\
        --model @('NOG/optimized/%s.model' % dataset)\
        --out @('NOG/refit/%s.model' % dataset)


if what == "crossmodel":
    '''
    1.    load the {dataset}
    2.    run the model against it
    '''
    mkdir -p NOG/crossmodel
    for model in fnames:
        if model != dataset:
            python biofilm/util/out.py --folds 0 @(loaddata)\
            --model @("NOG/refit/%s.model.model" % model) --out @('NOG/crossmodel/%s%s' % (model,dataset))



