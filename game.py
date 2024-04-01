import numpy as np
from numpy.linalg import norm
from scipy import optimize
def arr2str(arr, sign='+', rep=False, preci=3): return np.array2string(arr, separator=',', formatter={'all': lambda x: f'{x:{sign}.{preci}e}'}).replace('\n', '').replace(' ', '') if not rep else np.array_repr(arr).replace('\n', '').replace(' ', '')
def normalize(vec, normalizer, axis): return vec/normalizer(vec, axis=axis, keepdims=True)
fresult, frecord, logrecord = 'data/result.log', 'data/record.log', 'data/record.csv'


class Game:
    syms = 'abcdefghjklmopqrtuvwyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    maxnit, maxsubnit, gTOL, bTOL, sglTOL, resol = 700000, 10000, 1e-6, 1e-5, 0.9, 1e-10
    maxdbarr, maxtansteplen, backward_steplen = 1e-1, 1e-1, 1e-1

    def __init__(self, Ns, Ni, Na, gamma, ua, Ta):
        self.Ns, self.Ni, self.Na, self.gamma, self.ua, self.Ta = Ns, Ni, Na, gamma, ua, Ta
        self.max_value = ua.max()/(1-gamma)
        self.record = []

    def solve(self, barr, policy, value):
        Ns, Ni, Na = self.Ns, self.Ni, self.Na
        nit, norsteplen, tansteplen = 0, 1e-4, Game.maxtansteplen
        Norm_tan_last = np.inf
        while True:
            subnit, opt_ave, opt_red, adam_m, adam_v = 0, 0, 1, np.zeros((Ns, Ni, Na)), np.zeros((Ns, Ni, Na))
            while True:
                projgrad, policy_bias, regret_bias, regret, sigmaU_jac, value_res = self.projgrad(barr, policy, value)
                policy_biasnorm, regret_biasnorm, projgrad_norm, value_resangle = [norm(item, np.inf, axis=-1).max(axis=-1) for item in [policy_bias, regret_bias, projgrad]]+[value_res.max(axis=0)-value_res.min(axis=0)]
                pd_unbiased, pg_zero, vres_opt, opt_value = ((policy_biasnorm < Game.gTOL) & (regret_biasnorm < Game.gTOL)).all(), (projgrad_norm < 1e-10).all(), (value_resangle < 1e-10).all(), np.array([regret_biasnorm.max(), projgrad_norm.max(), value_resangle.max()])
                opt_ave, opt_red, opt_ocil = (lambda _red: (opt_value+0.9*_red, _red, opt_red*_red))(opt_ave-opt_value)
                norsteplen *= np.select([(opt_ocil[0] < 0), (opt_red[0] > 0)], [1-1e-2, 1+1e-4], default=1)
                adam_m, adam_v = (lambda exp_projgrad: (adam_m+1e-1*(exp_projgrad-adam_m), adam_v+1e-3*(exp_projgrad**2-adam_v)))(projgrad/policy)
                dpolicy = norsteplen*adam_m/(adam_v+Game.gTOL)**0.5
                # self.record.append(np.hstack([barr.reshape((Ns, Ni*Na)), policy.reshape((Ns, Ni*Na)), regret.reshape((Ns, Ni*Na)), value]).flatten())
                # recstr = f"|{subnit:^5d}|{'|'.join([arr2str(item) for item in [policy_biasnorm,regret_biasnorm,projgrad_norm,value_resangle,norsteplen]])}|\n"
                # print(recstr, end="")
                # with open(frecord, 'a') as fio:
                #     fio.writelines(recstr)
                if (nit := nit+1) > Game.maxnit:
                    # np.savetxt(logrecord, np.vstack(self.record), delimiter=',')
                    return (barr, policy, value), False, nit, [bp_bias, Collin_tan, Ori_tan, Norm_tan, policy_biasnorm, regret_biasnorm, projgrad_norm, value_resangle, norsteplen, tansteplen]
                elif (subnit := subnit+1) > Game.maxsubnit or pg_zero or (subnit > 8000 and pd_unbiased) or np.abs(opt_red[0]) < 1e-20 or norsteplen < 1e-12 and vres_opt:
                    if pd_unbiased:
                        if (norm((bp_bias := barr-policy*np.sum(barr, axis=-1, keepdims=True)), np.inf, axis=-1) < Game.bTOL).all():
                            # np.savetxt(logrecord, np.vstack(self.record), delimiter=',')
                            return (barr, policy, value), True, nit, [bp_bias, Collin_tan, Ori_tan, Norm_tan, policy_biasnorm, regret_biasnorm, projgrad_norm, value_resangle, norsteplen, tansteplen]
                    else:
                        barr, policy, value, sigmaU_jac = bpv_bkp
                    tansteplen = (tansteplen*np.select([~pd_unbiased, pg_zero], [0.5, 1.1], default=1.05)).clip(max=Game.maxtansteplen)
                    break
                policy = normalize(np.exp(np.log(policy)-dpolicy), normalizer=np.sum, axis=-1)
                value += 1e-1*value_res

            exp_tanvec, tanvec = self.tanvec(barr, policy, sigmaU_jac)
            sgl_decom = (lambda sgl_decom: (sgl_decom[0]**2, sgl_decom[1]))(np.linalg.svd(tanvec.reshape((Ns, Ni*Na, Ni*Na)).swapaxes(-1, -2))[1:])
            sgl_value, sgl_vec = (lambda index: [item[np.arange(Ns), index] for item in sgl_decom])(sgl_decom[0].argmax(axis=1))
            Collin_tan = (sgl_value/((tanvec**2).sum(axis=(1, 2, 3))))**0.5
            Norm_tan = np.abs(np.einsum('sxia,sx->sia', tanvec, sgl_vec))
            Ori_tan = Norm_tan/norm(tanvec, axis=1)
            sgl_encounter = (Collin_tan > Game.sglTOL) & (Ori_tan > 0.9*Game.sglTOL).all(axis=(-1, -2)) & (Norm_tan > Norm_tan_last).all(axis=(-1, -2))
            Norm_tan_last = Norm_tan
            sgl_avoidance = sgl_encounter.any() or (tansteplen < Game.maxtansteplen*1e0)

            dbarr_alpha = (tansteplen/norm(tanvec.sum(axis=(-1, -2)), axis=-1).max()).clip(max=Game.maxdbarr*tansteplen/Game.maxtansteplen)
            barr_next = ((1-dbarr_alpha)*barr+Game.backward_steplen*int(sgl_avoidance)*policy).clip(min=Game.resol)

            bpv_bkp = barr.copy(), policy.copy(), value.copy(), sigmaU_jac.copy()
            # recstr = f"|{nit:^7d}|{'|'.join([arr2str(item) for item in [barr_next, bp_bias, Collin_tan, Ori_tan, Norm_tan, policy_biasnorm, regret_biasnorm, projgrad_norm, value_resangle, norsteplen, tansteplen]])}|\n"
            # with open(frecord, 'a') as fio:
            #     fio.writelines(recstr)
            # with open(fresult, 'a') as fio:
            #     fio.writelines(f"barr,policy,value=np.{arr2str(barr,rep=True)},np.{arr2str(policy,rep=True)},np.{arr2str(value,rep=True)}\n")

            barr, dbarr = barr_next, 1-barr_next/barr
            policy = normalize(np.exp(np.log(policy)-np.einsum('siakl,skl->sia', exp_tanvec, dbarr)), normalizer=np.sum, axis=-1)

    def projgrad(self, barr, policy, value):
        Ns, Ni, Na, gamma, ua, Ta = self.Ns, self.Ni, self.Na, self.gamma, self.ua, self.Ta
        ustatic = ua+gamma*np.dot(Ta, value+self.max_value)
        sigmaU_jac = (lambda policy_: np.block([[np.einsum(f"{Game.syms[:Ni]}s{''.join([f',s{k}' for k in Game.syms[:Ni].replace(Game.syms[i],'').replace(Game.syms[j],'')])}->s{Game.syms[i]}{Game.syms[j]}", ustatic[..., i], *policy_[(np.arange(Ni) != i) & (np.arange(Ni) != j)]) if j != i else np.zeros((Ns, Na, Na)) for j in range(Ni)] for i in range(Ni)]))(policy.swapaxes(0, 1))
        sigmaU = np.einsum('sab,sb->sa', sigmaU_jac, policy.reshape((Ns, Ni*Na))).reshape((Ns, Ni, Na))/(Ni-1)
        _sigmaU_max, _sigmaU = (lambda _sigmaU_max: (_sigmaU_max, _sigmaU_max[..., np.newaxis]-sigmaU))(sigmaU.max(axis=-1))
        value_static = np.array([[optimize.root_scalar(lambda v: (barr[s, i]/(v+_sigmaU[s, i])).sum()-1, bracket=np.array([0, 1000]), method='brentq').root for i in range(Ni)] for s in range(Ns)])+_sigmaU_max
        regret = value_static[..., np.newaxis]-sigmaU
        value_res = value_static-np.einsum('sia,sia->si', policy, regret)-value
        # value_res = _sigmaU_max-value
        policy_bias, regret_bias = policy-barr/regret, regret-barr/policy
        scale_factor = 1+barr/(policy*regret)
        projgrad = (lambda grad: grad-(grad.sum(axis=-1)/Na)[..., np.newaxis])(regret_bias*scale_factor-np.einsum('sab,sa->sb', sigmaU_jac, (policy_bias*scale_factor).reshape((Ns, Ni*Na))).reshape((Ns, Ni, Na)))
        return projgrad, policy_bias, regret_bias, regret, sigmaU_jac, value_res

    def tanvec(self, barr, policy, sigmaU_jac):
        Ns, Ni, Na = self.Ns, self.Ni, self.Na
        comat1, comat2 = (lambda policy_: (np.eye(Ni*Na)*barr.reshape((Ns, Ni*Na))[..., np.newaxis, :]-sigmaU_jac*policy_[..., np.newaxis, :]*policy_[..., :, np.newaxis], np.kron(np.eye(Ni), np.ones(Na))*policy_[..., np.newaxis, :]))(policy.reshape((Ns, Ni*Na)))
        comat = np.block([[comat1, comat2.swapaxes(-1, -2)], [comat2, np.zeros((Ns, Ni, Ni))]])
        exp_tanvec = np.linalg.solve(comat, np.vstack([np.eye(Ni*Na), np.zeros((Ni, Ni*Na))])[np.newaxis, ...])[..., :Ni*Na, :].reshape((Ns, Ni, Na, Ni, Na))*barr[:, np.newaxis, np.newaxis, :, :]
        tanvec = (exp_tanvec*policy[..., np.newaxis, np.newaxis]).reshape((Ns, Ni*Na, Ni, Na))
        return exp_tanvec, tanvec
