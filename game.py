import numpy as np
from numpy.linalg import norm
from scipy import optimize
from multiprocessing import Pool
# np.set_printoptions(threshold=np.inf, linewidth=np.inf)
np.seterr(divide='ignore')
def arr2str(arr, sign='+', rep=False, preci=3): return np.array2string(arr, separator=',', formatter={'all': lambda x: f'{x:{sign}.{preci}e}'}).replace('\n', '').replace(' ', '') if not rep else np.array_repr(arr).replace('\n', '').replace(' ', '')
def normalize(vec, normalizer, axis): return vec/normalizer(vec, axis=axis, keepdims=True)


class Game:
    syms = 'abcdefghjklmopqrtuvwyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    maxnit, maxsubnit, gTOL, bTOL = 500000, 10000, 1e-6, 1e-5
    beta, eta, beta_decay = 2e-1, 2e-1, 0.9

    def __init__(self, Ns, Ni, Na, gamma, ua, Ta):
        self.Ns, self.Ni, self.Na, self.gamma, self.ua, self.Ta = Ns, Ni, Na, gamma, ua, Ta
        self.max_value = ua.max()/(1-gamma)
        self.record = []

    def solve(self, barr, policy, value):
        Ns, Ni, Na = self.Ns, self.Ni, self.Na
        nit, norsteplen, beta = 0, 1e-4, Game.beta*np.ones(1)
        while True:
            subnit, opt_ave, opt_red, adam_m, adam_v = 0, 0, 1, np.zeros((Ns, Ni, Na)), np.zeros((Ns, Ni, Na))
            while True:
                projgrad, policy_bias, regret_bias, regret, dp_res, piU_jac, cano_sect = self.onto_equilbundl(barr, policy, value)
                record_list = [norm(item, np.inf, axis=-1).max(axis=-1) for item in [cano_sect, policy_bias, regret_bias, projgrad]]+[dp_res.max(axis=0)-dp_res.min(axis=0), beta]
                cano_sect_norm, policy_biasnorm, regret_biasnorm, projgrad_norm, value_resangle, _ = record_list
                cano_sect_zero, pd_unbiased, pg_zero, vres_opt, opt_value = (cano_sect_norm < Game.bTOL).all(), ((policy_biasnorm < Game.gTOL) & (regret_biasnorm < Game.gTOL)).all(), (projgrad_norm < 1e-10).all(), (value_resangle < 1e-10).all(), np.array([regret_biasnorm.max(), projgrad_norm.max(), value_resangle.max()])
                opt_ave, opt_red, opt_ocil = (lambda _red: (opt_value+0.9*_red, _red, opt_red*_red))(opt_ave-opt_value)
                norsteplen *= np.select([(opt_ocil[0] < 0), (opt_red[0] > 0)], [1-1e-2, 1+1e-4], default=1)
                adam_m, adam_v = (lambda exp_projgrad: (adam_m+1e-1*(exp_projgrad-adam_m), adam_v+1e-3*(exp_projgrad**2-adam_v)))(projgrad/policy)
                dpolicy = norsteplen*adam_m/(adam_v+Game.gTOL)**0.5
                # self.record.append(np.hstack([barr.reshape((Ns, Ni*Na)), policy.reshape((Ns, Ni*Na)), regret.reshape((Ns, Ni*Na)), value]).flatten())
                # recstr = f"|{subnit:^5d}|{'|'.join([arr2str(item) for item in record_list])}|\n"
                # print(recstr, end="")
                # with open(frecord, 'a') as fio:
                #     fio.writelines(recstr)
                if (nit := nit+1) > Game.maxnit:
                    # np.savetxt(logrecord, np.vstack(self.record), delimiter=',')
                    return (barr, policy, value), False, nit, record_list
                elif (subnit := subnit+1) > Game.maxsubnit or pg_zero or (subnit > 8000 and pd_unbiased) or np.abs(opt_red[0]) < 1e-20 or norsteplen < 1e-12 and vres_opt:
                    if pd_unbiased:
                        beta *= Game.beta_decay
                        if cano_sect_zero:
                            # np.savetxt(logrecord, np.vstack(self.record), delimiter=',')
                            return (barr, policy, value), True, nit, record_list
                    else:
                        beta += Game.beta
                        barr, policy, value, piU_jac = bpv_bkp
                    break
                policy = normalize(np.exp(np.log(policy)-dpolicy), normalizer=np.sum, axis=-1)
                value += 1e-1*dp_res

            diff = self.along_equilbundl(barr, policy, piU_jac)
            barr_next = (1-Game.eta)*cano_sect+beta*policy

            bpv_bkp = barr.copy(), policy.copy(), value.copy(), piU_jac.copy()
            recstr = f"|{nit:^7d}|{'|'.join([arr2str(item) for item in record_list])}|\n"
            with open(frecord, 'a') as fio:
                fio.writelines(recstr)

            barr, dbarr = barr_next, 1-barr_next/barr
            policy = normalize(np.exp(np.log(policy)-np.einsum('siakl,skl->sia', diff, dbarr)), normalizer=np.sum, axis=-1)

    def onto_equilbundl(self, barr, policy, value):
        Ns, Ni, Na, gamma, ua, Ta = self.Ns, self.Ni, self.Na, self.gamma, self.ua, self.Ta
        ustatic = ua+gamma*np.dot(Ta, value+self.max_value)
        piU_jac = (lambda _policy: np.block([[np.einsum(f"{Game.syms[:Ni]}s{''.join([f',s{k}' for k in Game.syms[:Ni].replace(Game.syms[i],'').replace(Game.syms[j],'')])}->s{Game.syms[i]}{Game.syms[j]}", ustatic[..., i], *_policy[(np.arange(Ni) != i) & (np.arange(Ni) != j)]) if j != i else np.zeros((Ns, Na, Na)) for j in range(Ni)] for i in range(Ni)]))(policy.swapaxes(0, 1))
        piU_vec = np.einsum('sab,sb->sa', piU_jac, policy.reshape((Ns, Ni*Na))).reshape((Ns, Ni, Na))/(Ni-1)
        _piU_vec_max, _cano_sect = (lambda _piU_vec_max: (_piU_vec_max, _piU_vec_max[..., np.newaxis]-piU_vec))(piU_vec.max(axis=-1))
        value_static = np.array([[optimize.root_scalar(lambda v: (barr[s, i]/(v+_cano_sect[s, i])).sum()-1, bracket=np.array([0, 1000]), method='brentq').root for i in range(Ni)] for s in range(Ns)])+_piU_vec_max
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

