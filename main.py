import numpy as np
from game import Game
from game_plot import GamePlot
from multiprocessing import Pool
np.seterr(divide='ignore')
fresult, frecord = 'data/result.log', 'data/record.log'
preci = 3
def arr2str(arr): return np.array2string(arr, separator=',', formatter={'all': lambda x: f'{x:.{preci}e}'}).replace('\n', '').replace(' ', '')


def test_static(seed):
    np.random.seed(seed)
    ua_static = np.random.rand(*((Na,)*Ni+(Ni,)))-0.5
    gamma = 0
    Ns = 1
    ua, Ta = ua_static[..., np.newaxis, :], np.ones((Na,)*Ni+(1, 1))
    np.random.seed()
    init_policy = np.random.dirichlet(np.ones(Na), size=(Nn, Ns, Ni))
    result, success, nit, record, record_csv = Game(Ns, Ni, Na, gamma, ua, Ta).solve(init_policy, frecord, verbose=1)
    with open(fresult, 'a') as fio:
        fio.writelines(f"|{seed:^4d}|{repr(success):7}|{nit:^7d}|{'|'.join([arr2str(item) for item in record])}|\n")
    cano_sect, policy, value = result


def test(seed):
    np.random.seed(seed)
    ua, Ta = np.random.rand(*((Na,)*Ni+(Ns, Ni)))-0.5, np.random.dirichlet(np.ones(Ns), size=(Na,)*Ni+(Ns,))
    np.random.seed()
    init_policy = np.random.dirichlet(np.ones(Na), size=(Nn, Ns, Ni))
    result, success, nit, record, record_csv = Game(Ns, Ni, Na, gamma, ua, Ta).solve(init_policy, frecord, verbose=1, plots=True)
    if record_csv:
        np.savetxt('data/record.csv', np.vstack(record_csv), delimiter=',')
    with open(fresult, 'a') as fio:
        fio.writelines(f"|{seed:^4d}|{repr(success):7}|{nit:^7d}|{'|'.join([arr2str(item) for item in record])}|\n")
    cano_sect, policy, value = result


def plottest(seed):
    np.random.seed(seed)
    ua, Ta = np.random.rand(*((Na,)*Ni+(Ns, Ni)))-0.5, np.random.dirichlet(np.ones(Ns), size=(Na,)*Ni+(Ns,))
    for nNn in range(Nn):
        gp = GamePlot(Ns, Ni, Na, gamma, ua, Ta, Nn, nNn, iter_per_frame=600)
        gp.anim()
    # gp.graph()


gamma = 0.5
Nn, Ns, Ni, Na = 6, 2, 2, 2
if __name__ == '__main__':
    print(f"Check results at {fresult} and iteration process at {frecord}.")
    open(fresult, 'w')
    with open(fresult, 'a') as fio:
        nlen = (6+preci)*Nn*Ns+(Ns-1)*Nn+2*Nn+Nn+1
        fio.writelines(f"|{'seed':^4}|{'success':^7}|{'nit':^7}|{'cano_sect':^{nlen}}|{'policy_bias':^{nlen}}|{'regret_bias':^{nlen}}|{'projgrad':^{nlen}}|{'dp_res':^{(6+preci)*Nn*Ni+(Ni-1)*Nn+2*Nn+Nn+1}}|{'beta':^{8+preci}}|\n")
    if single_test := True:
        seed = 1
        test(seed)
        if Ns == 2 and Ni == 2 and Na == 2:
            plottest(seed)
    else:
        with Pool(processes=8) as pool:
            pool.map(test, np.random.randint(2000, size=8), chunksize=1)
