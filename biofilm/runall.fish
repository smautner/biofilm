set -x 'MKL_NUM_THREADS' 1
set -x 'NUMBA_NUM_THREADS' 1
set -x 'OMP_NUM_THREADS' 1
set -x 'OPENBLAS_NUM_THREADS' 1


set lol -j 10 python biofilm-optimize.py --infile DATA.npz --method ExtraTrees
parallel $lol --randinit {1} --out res/{1} ::: (seq 10)





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

aaa

