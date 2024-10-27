import numpy as np
import torch
if not (use_torch := False):
    from game import Game
    np.seterr(divide='ignore', invalid='ignore')
else:
    from game_torch import Game
    torch.set_default_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_default_dtype(torch.float64)
from game_plot import GamePlot
from multiprocessing import Pool
fresult, frecord = 'data/result.log', 'data/record.log'
preci = 3
def arr2str(arr, preci): return np.array2string(arr, separator=',', formatter={'all': lambda x: f'{x:.{preci}e}'}).replace('\n', '').replace(' ', '')


def test(seed, test_static=False, verbose=0):
    global gamma
    np.random.seed(seed)
    ua, Ta = np.random.rand(*((Na,)*Ni+(Ns, Ni)))-0.5, np.random.dirichlet(np.ones(Ns), size=(Na,)*Ni+(Ns,))
    # np.random.seed()
    init_policy = np.random.dirichlet(np.ones(Na), size=(Nn, Ns, Ni))
    if test_static:
        gamma = 0
        Ta = np.kron(np.ones((Na,)*Ni)[..., np.newaxis, np.newaxis], np.eye(Ns))
    if use_torch:
        ua, Ta, init_policy = [torch.from_numpy(item).to(torch.get_default_device()) for item in [ua, Ta, init_policy]]
    result, success, nit, record, record_csv = Game(Ns, Ni, Na, gamma, ua, Ta).solve(init_policy, verbose=verbose, record_file=frecord, preci=preci, plots=True)
    if use_torch:
        result, record = [item.numpy() for item in result], [item.numpy() for item in record]
    if record_csv:
        np.savetxt('data/record.csv', np.vstack(record_csv) if not use_torch else torch.vstack(record_csv).numpy(), delimiter=',')
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
Nn, Ns, Ni, Na = 1, 3, 3, 3
if __name__ == '__main__':
    print(f"Check results at {fresult} and iteration process at {frecord}.")
    open(fresult, 'w')
    with open(fresult, 'a') as fio:
        nlen = (6+preci)*Nn*Ns+(Ns-1)*Nn+2*Nn+Nn+1
        fio.writelines(f"|{'seed':^4}|{'success':^7}|{'nit':^7}|{'cano_sect':^{nlen}}|{'policy_bias':^{nlen}}|{'regret_bias':^{nlen}}|{'projgrad':^{nlen}}|{'dp_res':^{(6+preci)*Nn*Ni+(Ni-1)*Nn+2*Nn+Nn+1}}|{'beta':^{8+preci}}|\n")
    if not (parallel_test := False):
        seed = 5
        test(seed, test_static=False, verbose=1)
        if Ns == 2 and Ni == 2 and Na == 2:
            plottest(seed)
    else:
        with Pool(processes=8) as pool:
            pool.map(test, range(2000), chunksize=1)
