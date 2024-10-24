import numpy as np
from numpy.linalg import norm
from scipy.optimize import root_scalar
preci = 3
def arr2str(arr): return np.array2string(arr, separator=',', formatter={'all': lambda x: f'{x:.{preci}e}'}).replace('\n', '').replace(' ', '')


class Game:
    syms = 'abcdefghjklmopqrtuvwyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    maxnit, maxsubnit, gTOL, bTOL = 500000, 10000, 1e-6, 1e-5
    beta, eta, beta_decay = 2e-1, 2e-1, 0.9
    """
    Approximates a perfect equilibrium of a given dynamic game.

    References:
    H. Sun, C. Xia, J. Tan, B. Yuan, X. Wang, and B. Liang, Geometric Structure and Polynomial-time Algorithm of Game Equilibria, 2024, https://arxiv.org/abs/2401.00747.
    """

    def __init__(self, Ns, Ni, Na, gamma, ua, Ta):
        """
        Parameters:
        Ns: Number of states.
        Ni: Number of players.
        Na: Number of actions.
        gamma: Discount factor.
        ua: Utility function.
        Ta: Transition function.
        """
        self.Ns, self.Ni, self.Na, self.gamma, self.ua, self.Ta = Ns, Ni, Na, gamma, ua, Ta
        self.max_value = ua.max()/(1-gamma)
        self.record = []

    def solve(self, barr, policy, value, record_file, verbose=0, plots=False):
        """
        Parameters:
        barr: Initial barrier parameter.
        policy: Initial policy.
        value: Initial value function.
        record_file: The file that the iteration process is output to.
        varbose: If 0, outputs nothing, if 1, outputs the outer iteration level, if 2, outputs both two iteration levels.
        plots: Whether to record every iteration step in a list, which can be used to animate the line search process later. (Only useful for 2-state 2-player 2-action dynamic games.)

        Returns:
        (cano_sect, policy, value): Resulting canonical section, policy, value function.
        True/False: Whether the program successfully converges to a perfect equilibrium.
        nit: Number of iteration steps.
        record_list=[cano_sect, policy_bias, regret_bias, projgrad, dp_res, beta]: Resulting norm of canonical section, (primal-dual) policy bias, (primal-dual) regret bias, projected gradient, dynamic programming residual, singular avoidance coefficient.
        self.record: The list recording every iteration step. (Empty if not recorded.)
        """
        Ns, Ni, Na = self.Ns, self.Ni, self.Na
        if verbose >= 1:
            open(record_file, 'w')
            with open(record_file, 'a') as fio:
                fio.writelines(f"|{'nit':^7}|{'cano_sect':^{(6+preci)*Ns+Ns+1}}|{'policy_bias':^{(6+preci)*Ns+Ns+1}}|{'regret_bias':^{(6+preci)*Ns+Ns+1}}|{'projgrad':^{(6+preci)*Ns+Ns+1}}|{'dp_res':^{(6+preci)*Ni+Ni+1}}|{'beta':^{6+preci+2}}|\n")
        nit, norsteplen, beta = 0, 1e-4, Game.beta*np.ones(1)
        while True:
            subnit, opt_ave, opt_red, adam_m, adam_v = 0, 0, 1, np.zeros((Ns, Ni, Na)), np.zeros((Ns, Ni, Na))
            while True:
                projgrad, policy_bias, regret_bias, regret, dp_res, piU_jac, cano_sect = self.onto_equilbundl(barr, policy, value)
                record_list = [norm(item, np.inf, axis=-1).max(axis=-1) for item in [cano_sect, policy_bias, regret_bias, projgrad]]+[dp_res.max(axis=0)-dp_res.min(axis=0), beta]

                cano_sect_norm, policy_biasnorm, regret_biasnorm, projgrad_norm, value_resangle, _ = record_list
                opt_ave, opt_red, opt_ocil = (lambda _red: (opt_ave-0.1*_red, _red, opt_red*_red))(opt_ave-regret_biasnorm.max())
                norsteplen *= np.select([(opt_ocil < 0), (opt_red > 0)], [1-1e-2, 1+1e-4], default=1)
                adam_m, adam_v = (lambda exp_projgrad: (adam_m+1e-1*(exp_projgrad-adam_m), adam_v+1e-3*(exp_projgrad**2-adam_v)))(projgrad/policy)
                dpolicy = norsteplen*adam_m/(adam_v+Game.gTOL)**0.5

                if plots and Ns == 2 and Ni == 2 and Na == 2:
                    self.record.append(np.hstack([barr.reshape((Ns, Ni*Na)), policy.reshape((Ns, Ni*Na)), regret.reshape((Ns, Ni*Na)), value]).flatten())
                if verbose >= 2:
                    with open(record_file, 'a') as fio:
                        fio.writelines(f"|{subnit:^5d}|{'|'.join([arr2str(item) for item in record_list])}|\n")

                if (nit := nit+1) > Game.maxnit:
                    return (cano_sect, policy, value), False, nit, record_list, self.record
                elif (subnit := subnit+1) > Game.maxsubnit or (projgrad_norm < 1e-10).all() or np.abs(opt_red) < 1e-20 or norsteplen < 1e-12 and (value_resangle < 1e-10).all():
                    if ((policy_biasnorm < Game.gTOL) & (regret_biasnorm < Game.gTOL)).all():
                        beta *= Game.beta_decay
                        if (cano_sect_norm < Game.bTOL).all():
                            return (cano_sect, policy, value), True, nit, record_list, self.record
                    else:
                        beta += Game.beta
                        barr, policy, value, piU_jac, cano_sect = bpv_bkp
                    break
                policy = (lambda vec: vec/np.sum(vec, axis=-1, keepdims=True))(np.exp(np.log(policy)-dpolicy))
                value += 1e-1*dp_res

            bpv_bkp = barr.copy(), policy.copy(), value.copy(), piU_jac.copy(), cano_sect.copy()
            if verbose >= 1:
                with open(record_file, 'a') as fio:
                    fio.writelines(f"|{nit:^7d}|{'|'.join([arr2str(item) for item in record_list])}|\n")

            diff = self.along_equilbundl(barr, policy, piU_jac)
            barr, dbarr = (lambda barr_next: (barr_next, 1-barr_next/barr))((1-Game.eta)*cano_sect+beta*policy)
            policy = (lambda vec: vec/np.sum(vec, axis=-1, keepdims=True))(np.exp(np.log(policy)-np.einsum('siakl,skl->sia', diff, dbarr)))

    def onto_equilbundl(self, barr, policy, value):
        Ns, Ni, Na, gamma, ua, Ta = self.Ns, self.Ni, self.Na, self.gamma, self.ua, self.Ta
        ustatic = ua+gamma*np.dot(Ta, value+self.max_value)
        piU_jac = (lambda _policy: np.block([[np.einsum(f"{Game.syms[:Ni]}s{''.join([f',s{k}' for k in Game.syms[:Ni].replace(Game.syms[i], '').replace(Game.syms[j], '')])}->s{Game.syms[i]}{Game.syms[j]}", ustatic[..., i], *_policy[(np.arange(Ni) != i) & (np.arange(Ni) != j)]) if j != i else np.zeros((Ns, Na, Na)) for j in range(Ni)] for i in range(Ni)]))(policy.swapaxes(0, 1))
        piU_vec = np.einsum('sab,sb->sa', piU_jac, policy.reshape((Ns, Ni*Na))).reshape((Ns, Ni, Na))/(Ni-1)
        _piU_vec_max, _cano_sect = (lambda _piU_vec_max: (_piU_vec_max, _piU_vec_max[..., np.newaxis]-piU_vec))(piU_vec.max(axis=-1))
        value_static = np.array([[root_scalar(lambda v: (barr[s, i]/(v+_cano_sect[s, i])).sum()-1, bracket=np.array([0, 1000]), method='brentq').root for i in range(Ni)] for s in range(Ns)])+_piU_vec_max
        regret = value_static[..., np.newaxis]-piU_vec
        policy_bias, regret_bias = policy-barr/regret, regret-barr/policy
        projgrad = (lambda grad: grad-(grad.sum(axis=-1)/Na)[..., np.newaxis])(regret_bias-np.einsum('sab,sa->sb', piU_jac, policy_bias.reshape((Ns, Ni*Na))).reshape((Ns, Ni, Na)))
        dp_res = value_static-np.einsum('sia,sia->si', policy, regret)-value
        return projgrad, policy_bias, regret_bias, regret, dp_res, piU_jac, policy*_cano_sect

    def along_equilbundl(self, barr, policy, piU_jac):
        Ns, Ni, Na = self.Ns, self.Ni, self.Na
        _policy = policy.reshape((Ns, Ni*Na))
        policy_barr = (policy/barr).reshape((Ns, Ni*Na))
        comat11 = np.eye(Ni*Na)-piU_jac*_policy[:, np.newaxis, :]*policy_barr[:, :, np.newaxis]
        comat21 = np.kron(np.eye(Ni), np.ones(Na))[np.newaxis, ...]*_policy[:, np.newaxis, :]
        comat12 = np.kron(np.eye(Ni), np.ones((Na, 1)))[np.newaxis, ...]*policy_barr[:, :, np.newaxis]
        comat = np.block([[comat11, comat12], [comat21, np.zeros((Ns, Ni, Ni))]])
        diff = np.linalg.solve(comat, np.vstack([np.eye(Ni*Na), np.zeros((Ni, Ni*Na))])[np.newaxis, ...])[..., :Ni*Na, :].reshape((Ns, Ni, Na, Ni, Na))
        return diff
