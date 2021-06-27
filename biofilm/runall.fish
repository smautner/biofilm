set -x 'MKL_NUM_THREADS' 1
set -x 'NUMBA_NUM_THREADS' 1
set -x 'OMP_NUM_THREADS' 1
set -x 'OPENBLAS_NUM_THREADS' 1

set folds (seq 0 4)
set randseeds (seq 0 3)

set prog -j 10 python biofilm-features.py 
set load  --infile lncRNA.npz  --foldselect {1} --Z True
set task --method svm --out res/o_{1}
parallel $prog $load $task ::: $folds 

set prog -j 5 --joblog log.txt python biofilm-optimize6.py 
set task --method sgd --out res/oo_{1} --featurefile res/o_{1}.npz --time 120 --n_jobs 6
parallel $prog $load $task ::: $folds 

python biofilm-out --infile res/*.csv --showproba 20 --rawproba False

# -<< ZOMG  >>-  

function aaa
    python -c "
import numpy as np
import os

def load(folder,loader):
    r= []
    for f in os.listdir(folder): 
            dim = f[:f.find('.')].split('_')
            myres = { f'd{i}':int(z) for i,z in enumerate(dim) } 
            myres['data'] = loader(f'{folder}/{f}')
            r.append(myres)
    return r 

r = load('./res', lambda x: np.load(x)['arr_0'])
r = [di['data'] for di in r]
print(np.mean(r))

#import matplotlib
#matplotlib.use('module://matplotlib-sixel')
#import matplotlib.pyplot as plt
#plt.plot(r)
#plt.show()
"
end 


