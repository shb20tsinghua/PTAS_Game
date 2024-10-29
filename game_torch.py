import torch
syms = 'abcdefghjklmopqrtuvwyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
torch.set_default_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
torch.set_default_dtype(torch.float64)
def arr2str(arr): return str(arr)[7:-1].replace('\n', '').replace(' ', '')


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
        self.model = gamma, torch.from_numpy(ua).to(torch.get_default_device()), torch.from_numpy(Ta).to(torch.get_default_device())
        self.max_value = ua.max()/(1-gamma)
        self.record = []

    def solve(self, policy, pdbias_TOL=1e-6, canosect_TOL=1e-5, resolu=1e-10, canosect_stepln=2e-1, singuavoi_stepln=2e-1, singuavoi_stepln_decay=0.9, projgrad_stepln=2e-4, dynaprogr_stepln=1e-1, maxnit=500000, maxsubnit=10000, verbose=0, record_file=None, preci=3, plots=False):
        """
        Parameters:
        policy: Initial policy.

        Returns:
        (cano_sect, policy, value): Resulting canonical section, policy, value function.
        True/False: Whether the program successfully converges to a perfect equilibrium.
        nit: Number of iteration steps.
        record_list=[cano_sect, policy_bias, regret_bias, projgrad, dp_res]: Resulting norm of canonical section, (primal-dual) policy bias, (primal-dual) regret bias, projected gradient, dynamic programming residual.
        np.vstack(self.record): The ndarray recording every iteration step. (Empty if not recorded.)
        """
        policy = torch.from_numpy(policy).to(torch.get_default_device())
        torch.set_printoptions(precision=preci, sci_mode=True)
        self.Nn = policy.shape[0]
        Nn, Ns, Ni, Na = self.Nn, self.Ns, self.Ni, self.Na
        barr, value = policy, torch.ones((Nn, Ns, Ni))*self.max_value
        verbose = verbose if record_file else 0
        if verbose >= 1:
            open(record_file, 'w')
            with open(record_file, 'a') as fio:
                nlen = (6+preci)*Nn*Ns+(Ns-1)*Nn+2*Nn+Nn+1
                fio.writelines(f"|{'nit':^7}|{'canosetol':^{6+preci}}|{'cano_sect':^{nlen}}|{'policy_bias':^{nlen}}|{'regret_bias':^{nlen}}|{'projgrad':^{nlen}}|{'dp_res':^{(6+preci)*Nn*Ni+(Ni-1)*Nn+2*Nn+Nn+1}}|{'singuavoi':^{6+preci}}|\n")
        nit, singu_avoi, pre_projgrad, _canosect_TOL = 0, singuavoi_stepln, torch.zeros((Nn, Ns, Ni, Na)), 1e-2
        while True:
            subnit, adam_m, adam_v = 0, torch.zeros((Nn, Ns, Ni, Na)), torch.zeros((Nn, Ns, Ni, Na))
            while True:
                projgrad, policy_bias, regret_bias, regret, dp_res, piU_jac, cano_sect = self.onto_equilbundl(barr, policy, value)
                record_list = [torch.linalg.norm(item, float('inf'), dim=-1).max(dim=-1).values for item in [cano_sect, policy_bias, regret_bias, projgrad]]+[dp_res.max(dim=1).values-dp_res.min(dim=1).values]
                cano_sect_norm, policy_biasnorm, regret_biasnorm, projgrad_norm, value_resangle = record_list

                if plots and Ns == 2 and Ni == 2 and Na == 2:
                    self.record.append(torch.cat([barr.reshape((Nn, Ns, Ni*Na)), policy.reshape((Nn, Ns, Ni*Na)), regret.reshape((Nn, Ns, Ni*Na)), value], dim=-1).flatten())
                if verbose >= 2:
                    with open(record_file, 'a') as fio:
                        fio.writelines(f"|{subnit:^7d}|{_canosect_TOL:^.{preci}e}|{'|'.join([arr2str(item) for item in record_list])}|{singu_avoi:^.{preci}e}|\n")

                if (nit := nit+1) > maxnit:
                    return [item.cpu().numpy() for item in [cano_sect, policy, value]], False, nit, [item.cpu().numpy() for item in record_list], torch.vstack(self.record).cpu().numpy() if self.record else None
                elif (subnit := subnit+1) > maxsubnit or torch.abs(pre_projgrad-projgrad).max() < 1e-12:
                    if (policy_biasnorm < pdbias_TOL).all() and (regret_biasnorm < pdbias_TOL).all() and (value_resangle < resolu).all():
                        singu_avoi *= singuavoi_stepln_decay
                        if (cano_sect_norm < _canosect_TOL).all():
                            if _canosect_TOL < canosect_TOL:
                                return [item.cpu().numpy() for item in [cano_sect, policy, value]], True, nit, [item.cpu().numpy() for item in record_list], torch.vstack(self.record).cpu().numpy() if self.record else None
                            else:
                                _canosect_TOL *= 0.9
                    else:
                        singu_avoi += singuavoi_stepln
                        barr, policy, value, piU_jac, cano_sect = bpv_bkp
                    break

                pre_projgrad = projgrad
                adam_m, adam_v = (lambda exp_projgrad: (adam_m+1e-1*(exp_projgrad-adam_m), adam_v+1e-3*(exp_projgrad**2-adam_v)))(projgrad/policy)
                dpolicy = projgrad_stepln*adam_m/(adam_v+4*torch.finfo(float).eps)**0.5
                policy = (lambda vec: vec/vec.sum(dim=-1, keepdims=True))(torch.exp(torch.log(policy)-torch.where(((policy_biasnorm < resolu) & (regret_biasnorm < resolu))[..., None, None], 0, dpolicy)))
                value += dynaprogr_stepln*dp_res

            bpv_bkp = barr.clone(), policy.clone(), value.clone(), piU_jac.clone(), cano_sect.clone()
            if verbose >= 1:
                with open(record_file, 'a') as fio:
                    fio.writelines(f"|{nit:^7d}|{_canosect_TOL:^.{preci}e}|{'|'.join([arr2str(item) for item in record_list])}|{singu_avoi:^.{preci}e}|\n")

            diff = self.along_equilbundl(barr, policy, piU_jac)
            barr, dbarr = (lambda barr_next: (barr_next, 1-barr_next/barr))(((1-canosect_stepln)*barr+singu_avoi*policy).clip(min=_canosect_TOL*0.2))
            policy = (lambda vec: vec/vec.sum(dim=-1, keepdims=True))(torch.exp(torch.log(policy)-torch.einsum('nsiakl,nskl->nsia', diff, dbarr)))

    def onto_equilbundl(self, barr, policy, value):
        Nn, Ns, Ni, Na = self.Nn, self.Ns, self.Ni, self.Na
        piU_jac, piU_vec = self.expected_util(policy, value)
        _piU_vec_max, _cano_sect = (lambda _piU_vec_max: (_piU_vec_max, _piU_vec_max[..., None]-piU_vec))(piU_vec.max(dim=-1).values)
        value_static = self._brentq(lambda v: (barr/(v[..., None]+_cano_sect)).sum(dim=-1)-1, torch.zeros((Nn, Ns, Ni)), torch.full((Nn, Ns, Ni), 1e3))+_piU_vec_max
        regret = value_static[..., None]-piU_vec
        policy_bias, regret_bias = policy-barr/regret, regret-barr/policy
        projgrad = (lambda grad: grad-(grad.sum(dim=-1)/Na)[..., None])(regret_bias-torch.einsum('nsab,nsa->nsb', piU_jac, policy_bias.reshape((Nn, Ns, Ni*Na))).reshape((Nn, Ns, Ni, Na)))
        dp_res = value_static-torch.einsum('nsia,nsia->nsi', policy, regret)-value
        return projgrad, policy_bias, regret_bias, regret, dp_res, piU_jac, policy*_cano_sect

    def along_equilbundl(self, barr, policy, piU_jac):
        Nn, Ns, Ni, Na = self.Nn, self.Ns, self.Ni, self.Na
        _policy = policy.reshape((Nn, Ns, Ni*Na))
        policy_barr = (policy/barr).reshape((Nn, Ns, Ni*Na))
        comat11 = torch.eye(Ni*Na)-piU_jac*_policy[:, :, None, :]*policy_barr[:, :, :, None]
        comat21 = torch.kron(torch.eye(Ni), torch.ones(Na))[None, ...]*_policy[:, :, None, :]
        comat12 = torch.kron(torch.eye(Ni), torch.ones((Na, 1)))[None, ...]*policy_barr[:, :, :, None]
        comat = torch.cat([torch.cat([comat11, comat12], dim=-1), torch.cat([comat21, torch.zeros((Nn, Ns, Ni, Ni))], dim=-1)], dim=-2)
        diff = torch.linalg.solve(comat, torch.cat([torch.eye(Ni*Na), torch.zeros((Ni, Ni*Na))], dim=-2)[None, None, ...])[..., :Ni*Na, :].reshape((Nn, Ns, Ni, Na, Ni, Na))
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
        ustatic = ua[..., None, :]+gamma*torch.tensordot(Ta, value+self.max_value, dims=([-1], [-2]))
        piU_jac = (lambda _policy: torch.cat([torch.cat([torch.einsum(f"{syms[:Ni]}sn{''.join([f',ns{k}' for k in syms[:Ni].replace(syms[i], '').replace(syms[j], '')])}->ns{syms[i]}{syms[j]}", ustatic[..., i], *_policy[(torch.arange(Ni) != i) & (torch.arange(Ni) != j)]) if j != i else torch.zeros((Nn, Ns, Na, Na)) for j in range(Ni)], dim=-1) for i in range(Ni)], dim=-2))(policy.swapaxes(1, 2).swapaxes(0, 1))
        piU_vec = torch.einsum('nsab,nsb->nsa', piU_jac, policy.reshape((Nn, Ns, Ni*Na))).reshape((Nn, Ns, Ni, Na))/(Ni-1)
        return piU_jac, piU_vec

    def _brentq(self, func, xa, xb, xtol=2e-12, rtol=4*torch.finfo(float).eps, maxiter=100):
        """
            Exactly the same as https://github.com/scipy/scipy/blob/main/scipy/optimize/Zeros/brentq.c
            See also https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html
            torch.set_default_dtype(torch.float64) is necessary for this function to work.
        """
        xshape = xa.shape
        xpre, fpre, xcur, fcur, xblk, fblk = xa, func(xa), xb, func(xb), torch.zeros(xshape), torch.zeros(xshape)
        spre, scur = torch.zeros(xshape), torch.zeros(xshape)
        for _ in range(maxiter):
            xblk, fblk, spre, scur = torch.where(((fpre != 0) & (fcur != 0) & (torch.sign(fpre) != torch.sign(fcur))), torch.stack([xpre, fpre, xcur-xpre, xcur-xpre]), torch.stack([xblk, fblk, spre, scur]))
            _condition = torch.abs(fblk) < torch.abs(fcur)
            xpre, xcur, fpre, fcur = (lambda xf_state: torch.where(_condition, xf_state[[1, 2, 4, 5]], xf_state[[0, 1, 3, 4]]))(torch.stack([xpre, xcur, xblk, fpre, fcur, fblk]))
            xblk, fblk = torch.where(_condition, torch.stack([xpre, fpre]), torch.stack([xblk, fblk]))
            delta, sbis = (xtol+rtol*torch.abs(xcur))/2, (xblk-xcur)/2
            if (conv_condition := (fcur == 0) | (torch.abs(sbis) < delta)).all():
                return xcur
            stry = torch.where(xpre == xblk, -fcur*(xcur-xpre)/(fcur-fpre), (lambda dpre, dblk: -fcur*(fblk*dblk-fpre*dpre)/(dblk*dpre*(fblk-fpre)))((fpre-fcur)/(xpre-xcur), (fblk-fcur)/(xblk-xcur)))
            spre, scur = torch.where(((torch.abs(spre) > delta) & (torch.abs(fcur) < torch.abs(fpre)) & (2*torch.abs(stry) < torch.minimum(torch.abs(spre), 3*torch.abs(sbis)-delta))), torch.stack([scur, stry]), torch.stack([sbis, sbis]))
            xpre, fpre = xcur, fcur
            xcur = xcur+torch.where(~conv_condition, torch.where(torch.abs(scur) > delta, scur, torch.where(sbis > 0, delta, -delta)), 0)
            fcur = func(xcur)
