import numpy as np
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
        self.Ns, self.Ni, self.Na = Ns, Ni, Na
        self.model = gamma, ua, Ta
        self.max_value = ua.max()/(1-gamma)
        self.record = []

    def solve(self, policy, canosect_TOL=1e-5, maxnit=500000, canosect_stepln=2e-1, singuavoi_stepln=2e-1, verbose=0, record_file=None, preci=3, plots=False):
        """
        Parameters:
        policy: Initial policy.

        Returns:
        [cano_sect, policy, value]: Resulting canonical section, policy, value function.
        True/False: Whether the program successfully converges to a perfect equilibrium.
        nit: Number of iteration steps.
        record_list=[cano_sect, policy_bias, projgrad, dp_res]: Resulting norm of canonical section, (primal-dual) policy bias, projected gradient, dynamic programming residual.
        np.vstack(self.record): The ndarray recording every iteration step. (None if not recorded.)
        """
        init_barr, projgrad_stepln, dynaprogr_stepln, tangent_stepln, maxsubnit = 1e3, 2e-4, 1e-1, 2e-2, 10000
        self.Nn = policy.shape[0]
        Nn, Ns, Ni, Na = self.Nn, self.Ns, self.Ni, self.Na
        barr, value = policy*init_barr, np.ones((Nn, Ns, Ni))*self.max_value
        verbose = verbose if record_file else 0
        if verbose >= 1:
            open(record_file, 'w')
            with open(record_file, 'a') as fio:
                nlen = (6+preci)*Nn*Ns+(Ns-1)*Nn+2*Nn+Nn+1
                fio.writelines(f"|{'nit':^7}|{'canosetol':^{6+preci}}|{'cano_sect':^{nlen}}|{'policy_bias':^{nlen}}|{'projgrad':^{nlen}}|{'dp_res':^{(6+preci)*Nn*Ni+(Ni-1)*Nn+2*Nn+Nn+1}}|{'singuavoi':^{6+preci}}|\n")
        nit, pre_projgrad, _canosect_TOL, _singuavoi = 0, 0, 1e-2, 1.0
        while True:
            subnit, adam_m, adam_v = 0, 0, 0
            while True:
                projgrad, policy_bias, regret, dp_res, comat11, cano_sect = self.onto_equilbundl(barr, policy, value)
                record_list = [np.linalg.norm(item, np.inf, axis=-1).max(axis=-1) for item in [cano_sect, policy_bias, projgrad]]+[dp_res.max(axis=1)-dp_res.min(axis=1)]
                cano_sect_norm, policy_biasnorm, projgrad_norm, value_resangle = record_list
                def unbiased(tol): return (policy_biasnorm < tol).all() and (value_resangle < tol).all()

                if plots and Ns == 2 and Ni == 2 and Na == 2:
                    self.record.append(np.block([barr.reshape((Nn, Ns, Ni*Na)), policy.reshape((Nn, Ns, Ni*Na)), regret.reshape((Nn, Ns, Ni*Na)), value]).flatten())
                if verbose >= 2:
                    print(f"|{subnit:^7d}|{_canosect_TOL:^.{preci}e}|{'|'.join([arr2str(item, preci) for item in record_list])}|{_singuavoi:^.{preci}e}|")

                if (nit := nit+1) > maxnit:
                    return [cano_sect, policy, value], False, nit, record_list, np.vstack(self.record) if self.record else None
                elif (subnit := subnit+1) > maxsubnit or unbiased(1e-9) or np.abs(pre_projgrad-(pre_projgrad := projgrad)).max() < 1e-12:
                    if unbiased(3e-8):
                        _singuavoi *= 0.2
                        if (cano_sect_norm < _canosect_TOL).all():
                            if _canosect_TOL < canosect_TOL:
                                return [cano_sect, policy, value], True, nit, record_list, np.vstack(self.record) if self.record else None
                            else:
                                _canosect_TOL *= 1-canosect_stepln
                    else:
                        if not unbiased(1e-6):
                            barr, policy, value, regret, comat11 = bpv_bkp
                        _singuavoi += 1
                    break

                adam_m += 1e-1*(projgrad-adam_m)
                adam_v += 1e-3*(projgrad**2-adam_v)
                dpolicy = projgrad_stepln*adam_m/(adam_v+4*np.finfo(float).eps)**0.5
                policy = (lambda vec: vec/vec.sum(axis=-1, keepdims=True))(np.exp(np.log(policy)-np.where((policy_biasnorm < 1e-10)[..., np.newaxis, np.newaxis], 0, dpolicy)))
                value += dynaprogr_stepln*dp_res

            bpv_bkp = barr.copy(), policy.copy(), value.copy(), regret.copy(), comat11.copy()
            diff = self.along_equilbundl(policy, regret, comat11)
            diff_ln = np.linalg.norm(policy[..., np.newaxis]*diff.sum(axis=-1), axis=-2).max(axis=-2)
            _canosect_stepln = (tangent_stepln/diff_ln).clip(max=canosect_stepln)/(1+_singuavoi)
            barr, dbarr = (lambda barr_next: (barr_next, 1-barr_next/barr))(((1-_canosect_stepln)[..., np.newaxis]*barr+_singuavoi*singuavoi_stepln*policy).clip(min=0.2*_canosect_TOL))
            policy = (lambda vec: vec/vec.sum(axis=-1, keepdims=True))(np.exp(np.log(policy)-np.einsum('nsiakl,nskl->nsia', diff, dbarr)))
            if verbose >= 1:
                with open(record_file, 'a') as fio:
                    fio.writelines(f"|{nit:^7d}|{_canosect_TOL:^.{preci}e}|{'|'.join([arr2str(item, preci) for item in record_list])}|{_singuavoi:^.{preci}e}|\n")

    def onto_equilbundl(self, barr, policy, value):
        Nn, Ns, Ni, Na = self.Nn, self.Ns, self.Ni, self.Na
        piU_jac, piU_vec = self.expected_util(policy, value)
        _piU_vec_max, _cano_sect = (lambda _piU_vec_max: (_piU_vec_max, _piU_vec_max[..., np.newaxis]-piU_vec))(piU_vec.max(axis=-1))
        value_static = self._brentq(lambda v: (barr/(v[..., np.newaxis]+_cano_sect)).sum(axis=-1)-1, np.zeros((Nn, Ns, Ni)), Na*barr.max(axis=-1))+_piU_vec_max
        regret = value_static[..., np.newaxis]-piU_vec
        policy_bias = policy-barr/regret
        comat11 = np.eye(Ni*Na)-piU_jac*policy.reshape((Nn, Ns, 1, Ni*Na))/regret.reshape((Nn, Ns, Ni*Na, 1))
        projgrad = (lambda grad: grad-grad.sum(axis=-1, keepdims=True)*policy)(np.einsum('nsab,nsa->nsb', comat11, (policy_bias*regret).reshape((Nn, Ns, Ni*Na))).reshape((Nn, Ns, Ni, Na)))
        dp_res = value_static-np.einsum('nsia,nsia->nsi', policy, regret)-value
        return projgrad, policy_bias, regret, dp_res, comat11, policy*_cano_sect

    def along_equilbundl(self, policy, regret, comat11):
        Nn, Ns, Ni, Na = self.Nn, self.Ns, self.Ni, self.Na
        comat21 = np.kron(np.eye(Ni), np.ones((1, Na)))*policy.reshape((Nn, Ns, 1, Ni*Na))
        comat12 = np.kron(np.eye(Ni), np.ones((Na, 1)))/regret.reshape((Nn, Ns, Ni*Na, 1))
        comat = np.block([[comat11, comat12], [comat21, np.zeros((Nn, Ns, Ni, Ni))]])
        diff = np.linalg.solve(comat, np.vstack([np.eye(Ni*Na), np.zeros((Ni, Ni*Na))])[np.newaxis, np.newaxis, ...])[..., :Ni*Na, :].reshape((Nn, Ns, Ni, Na, Ni, Na))
        return diff

    def expected_util(self, policy, value):
        """
            Implement the algorithm of the expected utility problem of your own model here.

            Below is our implementation based on normal form representation of games.
            However, note that the method require only piU_jac and piU_vec to perform the numerical computations, which is a variant of the expected utility problem that has polynomial-time algorithms in many other models.

            The method applies to succint representations of games, such as graphical games, action-graph games, polymatrix games, and anonymous games.
            See more about succint games at https://en.wikipedia.org/wiki/Succinct_game

            The method also applies to model-free cases, where piU_jac and piU_vec are estimated from samples as an agent interacts with a game instance.
        """
        Nn, Ns, Ni, Na = self.Nn, self.Ns, self.Ni, self.Na
        gamma, ua, Ta = self.model
        ustatic = ua[..., np.newaxis, :]+gamma*np.dot(Ta, value+self.max_value)
        syms = 'abcdefghjklmopqrtuvwyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        piU_jac = (lambda _policy: np.block([[np.einsum(f"{syms[:Ni]}sn{''.join([f',ns{k}' for k in syms[:Ni].replace(syms[i], '').replace(syms[j], '')])}->ns{syms[i]}{syms[j]}", ustatic[..., i], *_policy[(np.arange(Ni) != i) & (np.arange(Ni) != j)]) if j != i else np.zeros((Nn, Ns, Na, Na)) for j in range(Ni)] for i in range(Ni)]))(policy.swapaxes(1, 2).swapaxes(0, 1))
        piU_vec = np.einsum('nsab,nsb->nsa', piU_jac, policy.reshape((Nn, Ns, Ni*Na))).reshape((Nn, Ns, Ni, Na))/(Ni-1)
        return piU_jac, piU_vec

    def _brentq(self, func, xa, xb, xtol=2e-12, rtol=4*np.finfo(float).eps, maxiter=100):
        """
            Exactly the same as https://github.com/scipy/scipy/blob/main/scipy/optimize/Zeros/brentq.c
            See also https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html
        """
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
