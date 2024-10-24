import numpy as np
from game import Game
from game_plot import GamePlot
from multiprocessing import Pool
np.seterr(divide='ignore')
fresult, frecord = 'data/result.log', 'data/record.log'
preci = 3
def arr2str(arr): return np.array2string(arr, separator=',', formatter={'all': lambda x: f'{x:.{preci}e}'}).replace('\n', '').replace(' ', '')


def test(seed):
    np.random.seed(seed)
    ua, Ta = np.random.rand(*((Na,)*Ni+(Ns, Ni)))-0.5, np.random.dirichlet(np.ones(Ns), size=(Na,)*Ni+(Ns,))
    result, success, nit, record, record_csv = Game(Ns, Ni, Na, gamma, ua, Ta).solve(np.ones((Ns, Ni, Na)), np.ones((Ns, Ni, Na))/Na, np.ones((Ns, Ni))*ua.max()/(1-gamma), frecord, verbose=1, plots=True)
    if record_csv:
        np.savetxt('data/record.csv', np.vstack(record_csv), delimiter=',')
    with open(fresult, 'a') as fio:
        fio.writelines(f"|{seed:^4d}|{repr(success):7}|{nit:^7d}|{'|'.join([arr2str(item) for item in record])}|\n")


def plottest(seed):
    np.random.seed(seed)
    ua, Ta = np.random.rand(*((Na,)*Ni+(Ns, Ni)))-0.5, np.random.dirichlet(np.ones(Ns), size=(Na,)*Ni+(Ns,))
    gp = GamePlot(Ns, Ni, Na, gamma, ua, Ta, iter_per_frame=200)
    gp.anim()
    # gp.graph()


gamma = 0.5
Ns, Ni, Na = 2, 2, 2
if __name__ == '__main__':
    print(f"Check results at {fresult} and iteration process at {frecord}.")
    open(fresult, 'w')
    with open(fresult, 'a') as fio:
        fio.writelines(f"|{'seed':^4}|{'success':^7}|{'nit':^7}|{'cano_sect':^{(6+preci)*Ns+Ns+1}}|{'policy_bias':^{(6+preci)*Ns+Ns+1}}|{'regret_bias':^{(6+preci)*Ns+Ns+1}}|{'projgrad':^{(6+preci)*Ns+Ns+1}}|{'dp_res':^{(6+preci)*Ni+Ni+1}}|{'beta':^{6+preci+2}}|\n")
    if single_test := True:
        seed = 0
        test(seed)
        if Ns == 2 and Ni == 2 and Na == 2:
            plottest(seed)
    else:
        with Pool(processes=8) as pool:
            pool.map(test, range(8), chunksize=1)
