import numpy as np
from multiprocessing import Pool
from game import Game
from gameplot import GamePlot
# np.set_printoptions(threshold=np.inf, linewidth=np.inf)
np.seterr(divide='ignore')
fresult, frecord, logrecord = 'data/result.log', 'data/record.log', 'data/record.csv'


def arr2str(arr, sign='+', rep=False, preci=3): return np.array2string(arr, separator=',', formatter={'all': lambda x: f'{x:{sign}.{preci}e}'}).replace('\n', '').replace(' ', '') if not rep else np.array_repr(arr).replace('\n', '').replace(' ', '')


def test(args):
    seed, bpv_init = (args, None) if type(args) == int else args
    np.random.seed(seed)
    ua, Ta = np.random.rand(*((Na,)*Ni+(Ns, Ni)))-0.5, np.random.dirichlet(np.ones(Ns), size=(Na,)*Ni+(Ns,))
    result, success, nit, record = Game(Ns, Ni, Na, gamma, ua, Ta).solve(*((bpv_init[0].reshape((Ns, Ni, Na)), bpv_init[1].reshape((Ns, Ni, Na)), bpv_init[2].reshape((Ns, Ni))) if bpv_init is not None else (np.ones((Ns, Ni, Na)), np.ones((Ns, Ni, Na))/Na, np.ones((Ns, Ni))*ua.max()/(1-gamma))))
    resultstr = f"|{seed:^4d}|{nit:^7d}|{repr(success):5}|{'|'.join([arr2str(item) for item in record])}|b{arr2str(result[0].flatten(),preci=6)}b|p{arr2str(result[1].flatten(),preci=6)}p|v{arr2str(result[2].flatten(),preci=6)}v|\n"
    print(resultstr)
    with open(fresult, 'a') as fio:
        fio.writelines(resultstr)


def plottest(seed):
    np.random.seed(seed)
    ua, Ta = np.random.rand(*((Na,)*Ni+(Ns, Ni)))-0.5, np.random.dirichlet(np.ones(Ns), size=(Na,)*Ni+(Ns,))
    gp = GamePlot(Ns, Ni, Na, gamma, ua, Ta, 27)
    gp.anim()
    gp.graph()


Ns, Ni, Na = 2, 2, 2
gamma = 0.5
if __name__ == '__main__':
    open(fresult, 'w'), open(frecord, 'w')
    if multiprocess_test := False:
        seed = 141
        test(seed)
        if Ns == 2 and Ni == 2 and Na == 2:
            plottest(seed)
    else:
        # with open('data/result0.log', 'r') as fio:
        #     records = fio.readlines()
        # tests = [(int(record[1:5]), [np.fromstring(record[record.find(f'|{mark}')+3:record.find(f'{mark}|')-1], sep=',') for mark in ['b', 'p', 'v']]) for record in records]
        with Pool(processes=8) as pool:
            pool.map(test, range(2000), chunksize=1)
