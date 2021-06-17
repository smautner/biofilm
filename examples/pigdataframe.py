
from numpy.random import Generator, PCG64
import dirtyopts
import pandas
import numpy as np
doc = '''
--pigdatasubsample int -1         # new arg to subsample
--randinit int -1                 # use biofilms randinit
--pigblacklist str no asser no genome kernel
'''

def read(filename): 
    args = dirtyopts.parse(doc)
    p = pandas.read_pickle(filename)
    # drop NAME 



    p.pop("NAME")
    if args.pigblacklist == 'genome':
        rm = np.where(p['GENOMEOVERLAP'] == 1)[0]
        p.drop(rm, inplace = True)
    elif args.pigblacklist == 'kernel':
        rm = np.where(p['KERNELNEIGH'] > .7)[0]
        p.drop(rm, inplace=True)
    
    p.reset_index(inplace=True)
    nid = np.where(p['CLASS'] == 0)[0].ravel()
    rg = Generator(PCG64(args.randinit))
    nid = rg.permuted(nid)
    nid = nid[args.pigdatasubsample:]




    if p.isnull().values.any():
        print(f"THERE ARE NANS")
    
    # shuffle and cut the nids then select the rows
    #p.drop(nid,inplace=True, axis = 0)
    p.drop(nid,inplace=True, axis = 0)
    
    # separate y and X 
    y = p.pop("CLASS").to_numpy()
    p.pop("GENOMEOVERLAP")
    p.pop("KERNELNEIGH")
    X = p.to_numpy()
    # now we load subsample and so on
    return X[:,1:],y # somehow dataframes attach an index so we jump one


if __name__ == "__main__":
    read("../biofilm/pigdataframe.pkl")
