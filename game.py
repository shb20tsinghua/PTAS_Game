import numpy as np
syms = 'abcdefghjklmopqrtuvwyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
def arr2str(arr, preci): return np.array2string(arr, separator=',', formatter={'all': lambda x: f'{x:.{preci}e}'}).replace('\n', '').replace(' ', '')


class Game:
    """
    Approximates perfect equilibria of a given dynamic game.

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

    def solve(self, policy, verbose=0, record_file=None, preci=3, plots=False, gTOL=1e-6, bTOL=1e-5, rTOL=1e-10, beta=2e-1, eta=2e-1, beta_decay=0.9, gradsteplen=2e-4, maxnit=500000, maxsubnit=10000):
        """
        Parameters:
        policy: Initial policy.
        varbose: If 0, outputs nothing, if 1, outputs the outer iteration level, if 2, outputs both two iteration levels.
        record_file: The file that the iteration process is output to.
        preci: Precision of the outputs.
        plots: Whether to record every iteration step in a list, which can be used to animate the line search process later. (Only useful for 2-state 2-player 2-action dynamic games.)

        Returns:
        (cano_sect, policy, value): Resulting canonical section, policy, value function.
        True/False: Whether the program successfully converges to a perfect equilibrium.
        nit: Number of iteration steps.
        record_list=[cano_sect, policy_bias, regret_bias, projgrad, dp_res, beta]: Resulting norm of canonical section, (primal-dual) policy bias, (primal-dual) regret bias, projected gradient, dynamic programming residual, singular avoidance coefficient.
        np.vstack(self.record): The ndarray recording every iteration step. (Empty if not recorded.)
        """
        self.Nn = policy.shape[0]
        Nn, Ns, Ni, Na = self.Nn, self.Ns, self.Ni, self.Na
        barr, value = policy, np.ones((Nn, Ns, Ni))*self.max_value
        verbose = verbose if record_file else 0
        if verbose >= 1:
            open(record_file, 'w')
            with open(record_file, 'a') as fio:
                nlen = (6+preci)*Nn*Ns+(Ns-1)*Nn+2*Nn+Nn+1
                fio.writelines(f"|{'nit':^7}|{'cano_sect':^{nlen}}|{'policy_bias':^{nlen}}|{'regret_bias':^{nlen}}|{'projgrad':^{nlen}}|{'dp_res':^{(6+preci)*Nn*Ni+(Ni-1)*Nn+2*Nn+Nn+1}}|{'beta':^{8+preci}}|\n")
        nit, betav = 0, beta*np.ones(1)
        while True:
            subnit, adam_m, adam_v = 0, np.zeros((Nn, Ns, Ni, Na)), np.zeros((Nn, Ns, Ni, Na))
            while True:
                projgrad, policy_bias, regret_bias, regret, dp_res, piU_jac, cano_sect = self.onto_equilbundl(barr, policy, value)
                _record_list = [np.linalg.norm(item, np.inf, axis=-1) for item in [cano_sect, policy_bias, regret_bias, projgrad]]
                record_list = [item.max(axis=-1) for item in _record_list]+[dp_res.max(axis=1)-dp_res.min(axis=1), betav]
                cano_sect_norm, policy_biasnorm, regret_biasnorm, projgrad_norm, value_resangle, _ = record_list

                if plots and Ns == 2 and Ni == 2 and Na == 2:
                    self.record.append(np.block([barr.reshape((Nn, Ns, Ni*Na)), policy.reshape((Nn, Ns, Ni*Na)), regret.reshape((Nn, Ns, Ni*Na)), value]).flatten())
                if verbose >= 2:
                    with open(record_file, 'a') as fio:
                        fio.writelines(f"|{subnit:^7d}|{'|'.join([arr2str(item, preci) for item in record_list])}|\n")

                if (nit := nit+1) > maxnit:
                    return [cano_sect, policy, value], False, nit, record_list, np.vstack(self.record) if self.record else None
                elif (subnit := subnit+1) > maxsubnit or (((pd_unbiased := ((policy_biasnorm < gTOL) & (regret_biasnorm < gTOL)).all()) and subnit > int(0.8*maxsubnit)) or (projgrad_norm < rTOL).all() and (value_resangle < rTOL).all()):
                    if pd_unbiased.all():
                        betav *= beta_decay
                        if (cano_sect_norm < bTOL).all():
                            return [cano_sect, policy, value], True, nit, record_list, np.vstack(self.record) if self.record else None
                    else:
                        betav += beta
                        barr, policy, value, piU_jac, cano_sect = bpv_bkp
                    break

                adam_m, adam_v = (lambda exp_projgrad: (adam_m+1e-1*(exp_projgrad-adam_m), adam_v+1e-3*(exp_projgrad**2-adam_v)))(projgrad/policy)
                dpolicy = gradsteplen*adam_m/(adam_v+rTOL)**0.5
                policy = (lambda vec: vec/vec.sum(axis=-1, keepdims=True))(np.exp(np.log(policy)-np.where(((policy_biasnorm < rTOL) & (regret_biasnorm < rTOL))[..., np.newaxis, np.newaxis], 0, dpolicy)))
                value += 1e-1*dp_res

            bpv_bkp = barr.copy(), policy.copy(), value.copy(), piU_jac.copy(), cano_sect.copy()
            if verbose >= 1:
                with open(record_file, 'a') as fio:
                    fio.writelines(f"|{nit:^7d}|{'|'.join([arr2str(item, preci) for item in record_list])}|\n")

            diff = self.along_equilbundl(barr, policy, piU_jac)
            barr, dbarr = (lambda barr_next: (barr_next, 1-barr_next/barr))(np.where((_record_list[0] < rTOL)[..., np.newaxis], barr, (1-eta)*cano_sect+betav*policy))
            policy = (lambda vec: vec/vec.sum(axis=-1, keepdims=True))(np.exp(np.log(policy)-np.einsum('nsiakl,nskl->nsia', diff, dbarr)))

    def onto_equilbundl(self, barr, policy, value):
        Nn, Ns, Ni, Na, gamma, ua, Ta = self.Nn, self.Ns, self.Ni, self.Na, self.gamma, self.ua, self.Ta
        ustatic = ua[..., np.newaxis, :]+gamma*np.dot(Ta, value+self.max_value)
        piU_jac = (lambda _policy: np.block([[np.einsum(f"{syms[:Ni]}sn{''.join([f',ns{k}' for k in syms[:Ni].replace(syms[i], '').replace(syms[j], '')])}->ns{syms[i]}{syms[j]}", ustatic[..., i], *_policy[(np.arange(Ni) != i) & (np.arange(Ni) != j)]) if j != i else np.zeros((Nn, Ns, Na, Na)) for j in range(Ni)] for i in range(Ni)]))(policy.swapaxes(1, 2).swapaxes(0, 1))
        piU_vec = np.einsum('nsab,nsb->nsa', piU_jac, policy.reshape((Nn, Ns, Ni*Na))).reshape((Nn, Ns, Ni, Na))/(Ni-1)
        _piU_vec_max, _cano_sect = (lambda _piU_vec_max: (_piU_vec_max, _piU_vec_max[..., np.newaxis]-piU_vec))(piU_vec.max(axis=-1))
        value_static = self._brentq(lambda v: (barr/(v[..., np.newaxis]+_cano_sect)).sum(axis=-1)-1, np.zeros((Nn, Ns, Ni)), np.full((Nn, Ns, Ni), 1e3))+_piU_vec_max
        regret = value_static[..., np.newaxis]-piU_vec
        policy_bias, regret_bias = policy-barr/regret, regret-barr/policy
        projgrad = (lambda grad: grad-(grad.sum(axis=-1)/Na)[..., np.newaxis])(regret_bias-np.einsum('nsab,nsa->nsb', piU_jac, policy_bias.reshape((Nn, Ns, Ni*Na))).reshape((Nn, Ns, Ni, Na)))
        dp_res = value_static-np.einsum('nsia,nsia->nsi', policy, regret)-value
        return projgrad, policy_bias, regret_bias, regret, dp_res, piU_jac, policy*_cano_sect

    def along_equilbundl(self, barr, policy, piU_jac):
        Nn, Ns, Ni, Na = self.Nn, self.Ns, self.Ni, self.Na
        _policy = policy.reshape((Nn, Ns, Ni*Na))
        policy_barr = (policy/barr).reshape((Nn, Ns, Ni*Na))
        comat11 = np.eye(Ni*Na)-piU_jac*_policy[:, :, np.newaxis, :]*policy_barr[:, :, :, np.newaxis]
        comat21 = np.kron(np.eye(Ni), np.ones(Na))[np.newaxis, ...]*_policy[:, :, np.newaxis, :]
        comat12 = np.kron(np.eye(Ni), np.ones((Na, 1)))[np.newaxis, ...]*policy_barr[:, :, :, np.newaxis]
        comat = np.block([[comat11, comat12], [comat21, np.zeros((Nn, Ns, Ni, Ni))]])
        diff = np.linalg.solve(comat, np.vstack([np.eye(Ni*Na), np.zeros((Ni, Ni*Na))])[np.newaxis, np.newaxis, ...])[..., :Ni*Na, :].reshape((Nn, Ns, Ni, Na, Ni, Na))
        return diff

    def _brentq(self, func, xa, xb, xtol=2e-12, rtol=4*np.finfo(float).eps, maxiter=100):
        xshape = xa.shape
        xpre, fpre, xcur, fcur, xblk, fblk = xa, func(xa), xb, func(xb), np.zeros(xshape), np.zeros(xshape)
        spre, scur = np.zeros(xshape), np.zeros(xshape)
        for _ in range(maxiter):
            xblk, fblk, spre, scur = np.where(((fpre != 0) & (fcur != 0) & (np.sign(fpre) != np.sign(fcur))), np.array([xpre, fpre, xcur-xpre, xcur-xpre]), np.array([xblk, fblk, spre, scur]))
            _condition = np.abs(fblk) < np.abs(fcur)
            xpre, xcur, fpre, fcur = (lambda xf_state: np.where(_condition, xf_state[[1, 2, 4, 5]], xf_state[[0, 1, 3, 4]]))(np.array([xpre, xcur, xblk, fpre, fcur, fblk]))
            xblk, fblk = np.where(_condition, np.array([xpre, fpre]), np.array([xblk, fblk]))
            delta, sbis = (xtol+rtol*np.abs(xcur))/2, (xblk-xcur)/2
            if (conv_condition := (fcur == 0) | (np.abs(sbis) < delta)).all():
                return xcur
            stry = np.where(xpre == xblk, -fcur*(xcur-xpre)/(fcur-fpre), (lambda dpre, dblk: -fcur*(fblk*dblk-fpre*dpre)/(dblk*dpre*(fblk-fpre)))((fpre-fcur)/(xpre-xcur), (fblk-fcur)/(xblk-xcur)))
            spre, scur = np.where(((np.abs(spre) > delta) & (np.abs(fcur) < np.abs(fpre)) & (2*np.abs(stry) < np.minimum(np.abs(spre), 3*np.abs(sbis)-delta))), np.array([scur, stry]), np.array([sbis, sbis]))
            xpre, fpre = xcur, fcur
            xcur = xcur+np.where(~conv_condition, np.where(np.abs(scur) > delta, scur, np.where(sbis > 0, delta, -delta)), 0)
            fcur = func(xcur)
