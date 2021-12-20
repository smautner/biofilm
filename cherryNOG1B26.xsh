import sys
from glob import glob
what =  sys.argv[1]
dataset =  sys.argv[2]



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

'''
1. load data
we can do it in a fancy way, by providing a python file that has a
read(path) function. as demonstrated in examples/cherriload.py
'''
loaddata = f'--infile NOG/data/{dataset} --loader examples/cherriload.py '.split()


'''
2. LETZ OPTYIMIZE
'''
if what == 'optimize':
    mkdir -p NOG/optimized
    loaddata += '--folds 0 --subsample 10000'.split()
    # TODO python -m biofilm.biofilm-optimize6 should work
    python biofilm/biofilm-optimize6.py  @(loaddata)\
        --out @(f'NOG/optimized/{dataset}') --preprocess True --n_jobs 30 --time 28800


'''
4. plot performance (so far)
'''
if what == 'plot1':
    for f in glob(f'NOG/optimized/*.csv'):
        python biofilm/biofilm-out.py --infiles  @(f)



'''
5. do crossval for all models
      use all 5 models to crossvalidate over all instances to compare them...
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




if what == 'trueplot':
    '''
    this is a leftover from the original scropt;
    i might want t o make it run with this in the future
    '''
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

