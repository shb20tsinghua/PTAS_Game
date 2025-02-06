import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
syms = 'abcdefghjklmopqrtuvwyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


class GamePlot:
    def __init__(self, Ns, Ni, Na, gamma, ua, Ta, Nn, nNn, iter_per_frame):
        self.nNn, self.Ns, self.Ni, self.Na, self.gamma, self.ua, self.Ta = nNn, Ns, Ni, Na, gamma, ua, Ta
        record = np.loadtxt('data/record.csv', delimiter=',')
        num = record.shape[0]
        frames = np.arange(num)[::iter_per_frame]
        frame_num = frames.shape[0]
        print(f"{num}/{iter_per_frame}={frame_num}")
        barr_f, policy_f, regret_f, value_f = (lambda rec: (rec[:, :, :Ni*Na].reshape((num, Ns, Ni, Na)), rec[:, :, Ni*Na:2*Ni*Na].reshape((num, Ns, Ni, Na)), rec[:, :, 2*Ni*Na:3*Ni*Na].reshape((num, Ns, Ni, Na)), rec[:, :, -Ni:]))(record.reshape((num, Nn, Ns, Ni+3*Ni*Na))[:, nNn, ...])
        self.curve_plot_data = self.curve_data(barr_f, policy_f, regret_f, value_f)
        self.barr, self.policy, self.regret, self.value = barr_f[frames], policy_f[frames], regret_f[frames], value_f[frames]
        self.num, self.frames, self.frame_num = num, frames, frame_num
        self.policy_f, self.regret_f, self.value_f = policy_f, regret_f, value_f

    def _get_cone_data(self, policy, value):
        Ns, Ni, Na, gamma, ua, Ta = self.Ns, self.Ni, self.Na, self.gamma, self.ua, self.Ta
        policy_ = np.einsum('nsia->insa', policy)
        upi = np.einsum(f"{syms[:Ni]}si,{','.join([f'ns{s}' for s in syms[:Ni]])}->nsi", ua, *policy_)
        I_gammaTpi = np.eye(Ns)-gamma*np.einsum(f"{syms[:Ni]}sx,{','.join([f'ns{s}' for s in syms[:Ni]])}->nsx", Ta, *policy_)
        res = -upi+np.einsum('nsx,nxi->nsi', I_gammaTpi, value)
        upi_ = np.array([np.einsum(f"{syms[:Ni]}s,{','.join([f'ns{k}' for k in syms[:Ni].replace(syms[i], '')])}->ns{syms[i]}", ua[..., i], *policy_[np.arange(Ni) != i]) for i in range(Ni)]).swapaxes(0, 1).swapaxes(1, 2)
        I_gammaTpi_ = np.array([np.eye(Ns)-gamma*np.einsum(f"{syms[:Ni]}sx,{','.join([f'ns{s}' for s in syms[:Ni].replace(syms[i], '')])}->n{syms[i]}sx", Ta, *policy_[np.arange(Ni) != i]) for i in range(Ni)]).swapaxes(0, 1)
        resb = -upi_+np.einsum('niasx,nxi->nsia', I_gammaTpi_, value)
        return upi, I_gammaTpi, res, upi_, I_gammaTpi_, resb

    def cone_data(self, policy, value):
        upi, I_gammaTpi, res, upi_, I_gammaTpi_, resb = self._get_cone_data(policy, value)
        apex = np.linalg.solve(I_gammaTpi, upi)
        apexb = np.linalg.solve(np.einsum('niabssx->niabsx', I_gammaTpi_[:, :, [[[0, 0], [0, 1]], [[1, 0], [1, 1]]], :, :]), np.einsum('nsiabs->niabs', upi_[..., [[[0, 0], [0, 1]], [[1, 0], [1, 1]]]])[..., np.newaxis])[..., 0].max(axis=(-2, -3)).swapaxes(-1, -2)
        cone_Y = value[:, :, np.newaxis, :]-res[:, np.newaxis, :, :]/(1-self.gamma)
        coneb_Y = value[:, :, np.newaxis, :]-resb.min(axis=-1)[:, np.newaxis, :, :]/(1-self.gamma)
        self.vaxlim = (lambda corner, length: np.array([corner-0.1*length, corner+1.1*length]))(cone_Y.min(axis=(0, 2)), (value.max(axis=0)-cone_Y.min(axis=(0, 2))).max(axis=0)[np.newaxis, :])
        vrange = np.linspace(*self.vaxlim, 100, axis=-1)
        apex_index = np.argmax(vrange[np.newaxis, :, :, :] >= apex[:, :, :, np.newaxis], axis=-1)
        apexb_index = np.argmax(vrange[np.newaxis, :, :, :] >= apexb[:, :, :, np.newaxis], axis=-1)
        cone_hyperp = np.array([(upi[:, 0, :, np.newaxis]-I_gammaTpi[:, 0, 1, np.newaxis, np.newaxis]*vrange[np.newaxis, 1, :, :])/I_gammaTpi[:, 0, 0, np.newaxis, np.newaxis],
                                (upi[:, 1, :, np.newaxis]-I_gammaTpi[:, 1, 0, np.newaxis, np.newaxis]*vrange[np.newaxis, 0, :, :])/I_gammaTpi[:, 1, 1, np.newaxis, np.newaxis]])
        coneb_hyperp = np.array([(upi_[:, 0, :, :, np.newaxis]-I_gammaTpi_[:, :, :, 0, 1, np.newaxis]*vrange[np.newaxis, 1, :, np.newaxis, :])/I_gammaTpi_[:, :, :, 0, 0, np.newaxis],
                                 (upi_[:, 1, :, :, np.newaxis]-I_gammaTpi_[:, :, :, 1, 0, np.newaxis]*vrange[np.newaxis, 0, :, np.newaxis, :])/I_gammaTpi_[:, :, :, 1, 1, np.newaxis]])
        V_DVdirec = np.array([value, cone_Y[:, [0, 1], [0, 1], :]]).swapaxes(0, 2)
        DVdirec_Y = np.array([cone_Y[:, [0, 1], [0, 1], np.newaxis, :][:, :, [0, 0], :], cone_Y]).swapaxes(0, 2)
        DV = value-res
        V_DhatVdirec = np.array([value, coneb_Y[:, [0, 1], [0, 1], :]]).swapaxes(0, 2)
        DhatVdirec_Yhat = np.array([coneb_Y[:, [0, 1], [0, 1], np.newaxis, :][:, :, [0, 0], :], coneb_Y]).swapaxes(0, 2)
        DhatV = value-resb.min(axis=-1)
        return apex, apexb, vrange, apex_index, apexb_index, cone_hyperp, coneb_hyperp, cone_Y, coneb_Y, value, V_DVdirec, DVdirec_Y, DV, V_DhatVdirec, DhatVdirec_Yhat, DhatV

    def cone_plots(self, fig, axes, legend_loc, legend_ncol, plot_traj=True):
        Ns, Ni, Na = self.Ns, self.Ni, self.Na
        [(axes[i].set_adjustable('box'), axes[i].set_aspect('equal')) for i in range(Ni)]
        [(axes[i].set_xlim(*self.vaxlim[:, 0, i]), axes[i].set_ylim(*self.vaxlim[:, 1, i])) for i in range(Ni)]
        [[axes[i].plot(*np.array([[self.vaxlim[0, s, i], self.vaxlim[0, s, i]], self.vaxlim[:, 1-s, i]])[[s, 1-s], :], 'k', clip_on=False) for i in range(Ni)] for s in range(Ns)]
        [[axes[i].plot(*np.array([self.vaxlim[0, s, i], self.vaxlim[1, 1-s, i]])[[s, 1-s], np.newaxis], f'k{["^", ">"][s]}', clip_on=False) for i in range(Ni)] for s in range(Ns)]
        [[axes[i].annotate(rf'$V_{s}^{i}$', np.array([self.vaxlim[0, s, i], self.vaxlim[1, 1-s, i]])[[s, 1-s]], xytext=(1-2*s, 2*s-1), textcoords='offset fontsize', fontweight='bold', ha='center', va='center') for i in range(Ni)] for s in range(Ns)]
        if plot_traj:
            [axes[i].plot(*self.value_f[:, :, i][1:].swapaxes(0, 1), 'C7.', markersize=1.0) for i in range(Ni)]
            del self.value_f
        cone_hyperp_plot = [[axes[i].plot([], [], 'r--', linewidth=0.8)[0] for i in range(Ni)] for s in range(Ns)]
        coneb_hyperp_plot = [[[axes[i].plot([], [], 'g:', linewidth=0.8)[0] for a in range(Na)] for i in range(Ni)] for s in range(Ns)]
        cone_plot = [[axes[i].plot([], [], 'r', linewidth=1.5, label=r'$C_\pi$')[0] for i in range(Ni)] for s in range(Ns)]
        coneb_plot = [[axes[i].plot([], [], 'g--', linewidth=1.5, label=r'$\hat{C}_\pi$')[0] for i in range(Ni)] for s in range(Ns)]
        cone_Y_plot = [[axes[i].plot([], [], 'rv', label=r'$Y_{xs}^i$')[0] for i in range(Ni)] for s in range(Ns)]
        coneb_Y_plot = [[axes[i].plot([], [], 'g^', label=r'$\hat{Y}_{xs}^i$')[0] for i in range(Ni)] for s in range(Ns)]
        onevec_plot = [axes[i].plot([], [], 'k', linewidth=1.0, label=r'$\mathbf{1}_s$')[0] for i in range(Ni)]
        coneb_apex_plot = [axes[i].plot([], [], 'C7.')[0] for i in range(Ni)]
        cone_apex_plot = [axes[i].plot([], [], 'b.', label=r'$V_{\pi s}^i$')[0] for i in range(Ni)]
        value_plot = [axes[i].plot([], [], 'ks', label=r'$V_s^i$')[0] for i in range(Ni)]
        V_DVdirec_plot = [axes[i].plot([], [], 'y', linewidth=1.0, label=r'$d_s^i$')[0] for i in range(Ni)]
        DVdirec_Y_plot = [[axes[i].plot([], [], 'b:', linewidth=1.0)[0] for i in range(Ni)] for s in range(Ns)]
        DV_plot = [axes[i].plot([], [], 'r*', label=r'$D_\pi(V_s^i)$')[0] for i in range(Ni)]
        V_DhatVdirec_plot = [axes[i].plot([], [], 'y--', linewidth=1.0, label=r'$\hat{d}_s^i$')[0] for i in range(Ni)]
        DhatVdirec_Yhat_plot = [[axes[i].plot([], [], 'b:', linewidth=1.0)[0] for i in range(Ni)] for s in range(Ns)]
        DhatV_plot = [axes[i].plot([], [], 'g*', label=r'$\hat{D}_\pi(V_s^i)$')[0] for i in range(Ni)]
        cano_sect_plot = [axes[i].plot([], [], 'b', linewidth=1.0, label=r'$\mathbf{1}_a\bar{\mu}_a^{si}(V_s^i,\pi_a^{si})$')[0] for i in range(Ni)]
        fig.legend(handles=[coneb_plot[0][0], cone_plot[0][0], cone_Y_plot[0][0], coneb_Y_plot[0][0], cone_apex_plot[0], value_plot[0], V_DVdirec_plot[0], DV_plot[0], V_DhatVdirec_plot[0], DhatV_plot[0], cano_sect_plot[0], onevec_plot[0]], loc=legend_loc, ncol=legend_ncol)
        return cone_hyperp_plot, coneb_hyperp_plot, cone_plot, coneb_plot, cone_Y_plot, coneb_Y_plot, cone_apex_plot, coneb_apex_plot, value_plot, onevec_plot, V_DVdirec_plot, DVdirec_Y_plot, DV_plot, V_DhatVdirec_plot, DhatVdirec_Yhat_plot, DhatV_plot, cano_sect_plot

    def cone_update(self, k, data, plots):
        Ns, Ni, Na = self.Ns, self.Ni, self.Na
        apex, apexb, vrange, apex_index, apexb_index, cone_hyperp, coneb_hyperp, cone_Y, coneb_Y, value, V_DVdirec, DVdirec_Y, DV, V_DhatVdirec, DhatVdirec_Yhat, DhatV = data
        cone_hyperp_plot, coneb_hyperp_plot, cone_plot, coneb_plot, cone_Y_plot, coneb_Y_plot, cone_apex_plot, coneb_apex_plot, value_plot, onevec_plot, V_DVdirec_plot, DVdirec_Y_plot, DV_plot, V_DhatVdirec_plot, DhatVdirec_Yhat_plot, DhatV_plot, cano_sect_plot = plots
        [[cone_hyperp_plot[s][i].set_data(*np.array([cone_hyperp[s, k, i, :], vrange[1-s, i]])[[s, 1-s]]) for i in range(Ni)] for s in range(Ns)]
        [[[coneb_hyperp_plot[s][i][a].set_data(*np.array([coneb_hyperp[s, k, i, a, :], vrange[1-s, i]])[[s, 1-s]]) for a in range(Na)] for i in range(Ni)] for s in range(Ns)]
        [[cone_plot[s][i].set_data(*np.array([cone_hyperp[s, k, i, apex_index[k, 1-s, i]:], vrange[1-s, i, apex_index[k, 1-s, i]:]])[[s, 1-s]]) for i in range(Ni)] for s in range(Ns)]
        [[coneb_plot[s][i].set_data(*np.array([coneb_hyperp[s, k, i, :, apexb_index[k, 1-s, i]:].max(axis=-2), vrange[1-s, i, apexb_index[k, 1-s, i]:]])[[s, 1-s]]) for i in range(Ni)] for s in range(Ns)]
        [[cone_Y_plot[s][i].set_data(*cone_Y[k, :, s, i, np.newaxis]) for i in range(Ni)] for s in range(Ns)]
        [[coneb_Y_plot[s][i].set_data(*coneb_Y[k, :, s, i, np.newaxis]) for i in range(Ni)] for s in range(Ns)]
        [cone_apex_plot[i].set_data(*apex[k, :,  i, np.newaxis]) for i in range(Ni)]
        [coneb_apex_plot[i].set_data(*apexb[k, :,  i, np.newaxis]) for i in range(Ni)]
        [value_plot[i].set_data(*value[k, :, i, np.newaxis]) for i in range(Ni)]
        [onevec_plot[i].set_data(*np.array([value[k, :, i], cone_Y[k, :, :, i].min(axis=-1)]).swapaxes(0, 1)) for i in range(Ni)]
        [V_DVdirec_plot[i].set_data(*V_DVdirec[:, k, :, i]) for i in range(Ni)]
        [[DVdirec_Y_plot[s][i].set_data(*DVdirec_Y[:, k, :, s, i]) for i in range(Ni)] for s in range(Ns)]
        [DV_plot[i].set_data(*DV[k, :, i, np.newaxis]) for i in range(Ni)]
        [V_DhatVdirec_plot[i].set_data(*V_DhatVdirec[:, k, :, i]) for i in range(Ni)]
        [[DhatVdirec_Yhat_plot[s][i].set_data(*DhatVdirec_Yhat[:, k, :, s, i]) for i in range(Ni)] for s in range(Ns)]
        [DhatV_plot[i].set_data(*DhatV[k, :, i, np.newaxis]) for i in range(Ni)]
        [cano_sect_plot[i].set_data(*np.array([DhatV[k, :, i], DV[k, :, i]]).swapaxes(0, 1)) for i in range(Ni)]

    def barrproblem_data(self, barr, policy, regret):
        init_barr = barr[0].sum(axis=-1)[0, 0]
        barr, regret = barr/init_barr, regret/init_barr
        dual_policy, dual_regret = barr/regret, barr/policy
        self.baxlim = np.maximum(regret.max(axis=(0, 2, 3)), dual_regret.max(axis=(0, 2, 3)))*1.1
        barr_uni, barr_invindex = np.unique(barr, return_inverse=True, axis=0)
        brange = np.exp(np.linspace(np.log(barr_uni), np.log(self.baxlim[np.newaxis, :, np.newaxis, np.newaxis]), 100, axis=-1))
        barr_hyperb = np.array([-brange, barr_uni[..., np.newaxis]/brange])
        rect = np.array([[np.stack([policy[..., 0], policy[..., 0], -dual_regret[..., 1], -dual_regret[..., 1], policy[..., 0]], axis=-1),
                          np.stack([policy[..., 1], -dual_regret[..., 0], -dual_regret[..., 0], policy[..., 1], policy[..., 1]], axis=-1)],
                         [np.stack([-regret[..., 1], -regret[..., 1], dual_policy[..., 0], dual_policy[..., 0], -regret[..., 1]], axis=-1),
                          np.stack([-regret[..., 0], dual_policy[..., 1], dual_policy[..., 1], -regret[..., 0], -regret[..., 0]], axis=-1)]])
        bias = np.array([[policy, dual_policy], [-regret[..., [1, 0]], -dual_regret[..., [1, 0]]]]).swapaxes(1, -1)
        return barr_hyperb, barr_invindex, rect, bias, regret

    def barrproblem_plots(self, fig, axes, legend_loc, legend_ncol, plot_traj=True):
        Ns, Ni, Na = axes.shape[0], self.Ni, self.Na
        [[(axes[s, i].set_adjustable('box'), axes[s, i].set_aspect('equal')) for i in range(Ni)] for s in range(Ns)]
        [[(axes[s, i].set_xlim(-self.baxlim[s], 1), axes[s, i].set_ylim(-self.baxlim[s], 1)) for i in range(Ni)] for s in range(Ns)]
        [[[axes[s, i].plot(*np.array([[-self.baxlim[s], 1], [0, 0]])[[a, 1-a], :], 'k') for a in range(Na)] for i in range(Ni)] for s in range(Ns)]
        [[[[axes[s, i].plot(*np.array([[-self.baxlim[s], 1], [0, 0]])[[a, 1-a], m], f'k{[["<", ">"], ["v", "^"]][a][m]}', clip_on=False) for a in range(Na)] for i in range(Ni)] for s in range(Ns)] for m in range(2)]
        [[[[axes[s, i].annotate([rf'$-r_{1-a}^{{{s}{i}}}$', rf'$\pi_{a}^{{{s}{i}}}$'][m], np.array([[-self.baxlim[s], 1], [0, 0]])[[a, 1-a], m], xytext=1.2*np.array([(1-2*a, 2*a-1), (-1, -1)])[m], textcoords='offset fontsize', fontweight='bold', ha='center', va='center') for a in range(Na)] for i in range(Ni)] for s in range(Ns)] for m in range(2)]
        [[(axes[s, i].plot(-self.baxlim[s], 0, 'k'), axes[s, i].plot([0, 0], [-self.baxlim[s], 1], 'k'), axes[s, i].plot([-self.baxlim[s], 1], [0, 0], 'k'), axes[s, i].plot([0, 0], [-self.baxlim[s], 1], 'k')) for i in range(Ni)] for s in range(Ns)]
        hyperp_plot = [[axes[s, i].plot([0, 1], [1, 0], 'orange', label=r'$\mathbf{1}_a\pi_a^i=\mathbf{1}^i$', zorder=0)[0] for i in range(Ni)] for s in range(Ns)]
        if plot_traj:
            [[axes[s, i].plot(*(-self.regret_f[:, s, i, :][:, [1, 0]]).swapaxes(0, 1), 'C7.', markersize=1.0) for i in range(Ni)] for s in range(Ns)]
            del self.regret_f
        hyperb_plot = [[[axes[s, i].plot([], [], 'blue', label=r'$\mu_a^i$')[0] for a in range(Na)] for i in range(Ni)] for s in range(Ns)]
        rect_plot = [[[axes[s, i].plot([], [], f'{["g", "r"][m]}--', label=[r'$\pi_a^i\circ \hat{r}_a^i=\mu_a^i$', r'$\hat{\pi}_a^i\circ r_a^i=\mu_a^i$'][m])[0] for i in range(Ni)] for s in range(Ns)] for m in range(2)]
        bias_plot = [[[axes[s, i].plot([], [], f'y{["^", "v"][m]}-', label=[r'$\pi_a^i-\hat{\pi}_a^i$', r'$r_a^i-\hat{r}_a^i$'][m])[0] for i in range(Ni)] for s in range(Ns)] for m in range(2)]
        dv_plot = [[axes[s, i].quiver(0, 0, 0.1, 0.1, scale=1, label=r'$dv^i$') for i in range(Ni)] for s in range(Ns)]
        dv_legend = [[axes[s, i].plot([], [], linestyle='', color='k', marker=r'$\longrightarrow$', markersize=15, label=r'$dv^i$')[0] for i in range(Ni)] for s in range(Ns)]
        fig.legend(handles=[rect_plot[0][0][0], rect_plot[1][0][0], hyperp_plot[0][0], hyperb_plot[0][0][0], bias_plot[0][0][0], bias_plot[1][0][0], dv_legend[0][0]], loc=legend_loc, ncol=legend_ncol)
        return hyperb_plot, rect_plot, bias_plot, dv_plot

    def barrproblem_update(self, k, data, plots):
        barr_hyperb, barr_invindex, rect, bias, regret = data
        hyperb_plot, rect_plot, bias_plot, dv_plot = plots
        Ns, Ni, Na = len(hyperb_plot), self.Ni, self.Na
        [[[hyperb_plot[s][i][a].set_data(*barr_hyperb[[1-a, a], barr_invindex[k], s, i, a, :]) for a in range(Na)] for i in range(Ni)] for s in range(Ns)]
        [[[rect_plot[m][s][i].set_data(*rect[m, :, k, s, i, :]) for i in range(Ni)] for s in range(Ns)] for m in range(2)]
        [[[bias_plot[m][s][i].set_data(*bias[m, :, k, s, i, :]) for i in range(Ni)] for s in range(Ns)] for m in range(2)]
        [[dv_plot[s][i].set_offsets(-regret[k, s, i, [1, 0]]) for i in range(Ni)] for s in range(Ns)]

    def kktcondition_data(self, ustatic, barr,  policy):
        def hyper_surface(p1, p2):
            u1 = -(p2*ustatic[:, 0, 0, :, 0, np.newaxis]+(1-p2)*ustatic[:, 0, 1, :, 0, np.newaxis])+(p2*ustatic[:, 1, 0, :, 0, np.newaxis]+(1-p2)*ustatic[:, 1, 1, :, 0, np.newaxis])
            u2 = -(p1*ustatic[:, 0, 0, :, 1, np.newaxis]+(1-p1)*ustatic[:, 1, 0, :, 1, np.newaxis])+(p1*ustatic[:, 0, 1, :, 1, np.newaxis]+(1-p1)*ustatic[:, 1, 1, :, 1, np.newaxis])
            hs1 = 0.5+0.5*(barr[..., 0, 0, np.newaxis]+barr[..., 0, 1, np.newaxis])/u1-0.5/u1*((u1-barr[..., 0, 0, np.newaxis]+barr[..., 0, 1, np.newaxis])**2+4*barr[..., 0, 0, np.newaxis]*barr[..., 0, 1, np.newaxis])**(0.5)
            hs2 = 0.5+0.5*(barr[..., 1, 0, np.newaxis]+barr[..., 1, 1, np.newaxis])/u2-0.5/u2*((u2-barr[..., 1, 0, np.newaxis]+barr[..., 1, 1, np.newaxis])**2+4*barr[..., 1, 0, np.newaxis]*barr[..., 1, 1, np.newaxis])**(0.5)
            return np.array([hs1, hs2])

        def tangent_vector(ustatic, barr,  policy, num):
            piU_jac = (lambda policy_: np.block([[np.einsum(f"n{syms[:Ni]}s{''.join([f',ns{k}' for k in syms[:Ni].replace(syms[i], '').replace(syms[j], '')])}->ns{syms[i]}{syms[j]}", ustatic[..., i], *policy_[(np.arange(Ni) != i) & (np.arange(Ni) != j)]) if j != i else np.zeros((num, Ns, Na, Na)) for j in range(Ni)] for i in range(Ni)]))(policy.swapaxes(1, 2).swapaxes(0, 1))
            comat1, comat2 = (lambda policy_: (np.eye(Ni*Na)*barr.reshape((num, Ns, Ni*Na))[..., np.newaxis, :]-piU_jac*policy_[..., np.newaxis, :]*policy_[..., :, np.newaxis], np.kron(np.eye(Ni), np.ones(Na))*policy_[..., np.newaxis, :]))(policy.reshape((num, Ns, Ni*Na)))
            comat = np.block([[comat1, comat2.swapaxes(-1, -2)], [comat2, np.zeros((num, Ns, Ni, Ni))]])
            exptan = np.linalg.solve(comat, np.hstack([np.eye(Ni*Na), np.zeros((Ni*Na, Ni))]).T[np.newaxis, np.newaxis, ...])[..., :Ni*Na, :].reshape((num, Ns, Ni, Na, Ni, Na))
            return exptan*policy[..., np.newaxis, np.newaxis]*barr[..., np.newaxis, np.newaxis, :, :]
        Ns, Ni, Na = barr.shape[1], self.Ni, self.Na
        prange = np.linspace(0, 1, 100)
        hypesurf = hyper_surface(prange, prange)
        dual_policy = (lambda hs: np.einsum('ains->nsia', np.array([hs, 1-hs])))(hyper_surface(policy[:, :, 0, 0, np.newaxis], policy[:, :, 1, 0, np.newaxis])[:, :, :, 0])
        rect = np.array([np.stack([policy[..., 0, 0], policy[..., 0, 0], dual_policy[..., 0, 0], dual_policy[..., 0, 0], policy[..., 0, 0]], axis=-1),
                         np.stack([policy[..., 1, 0], dual_policy[..., 1, 0], dual_policy[..., 1, 0], policy[..., 1, 0], policy[..., 1, 0]], axis=-1)])
        policy_bias = np.array([policy[..., 0], dual_policy[..., 0]]).swapaxes(0, -1)
        unbiased_index, unbiased_invindex = np.unique(barr[::-1], return_index=True, return_inverse=True, axis=0)[1:]
        unbiased_index, unbiased_invindex = barr.shape[0]-1-unbiased_index, unbiased_invindex[::-1]
        unbiased_policy = policy[unbiased_index]
        tanvec = tangent_vector(ustatic[unbiased_index], barr[unbiased_index],  unbiased_policy, unbiased_index.shape[0])
        return prange, hypesurf, rect, policy_bias, unbiased_policy, unbiased_invindex, tanvec

    def kktcondition_plots(self, fig, axes, legend_loc, legend_ncol, plot_traj=True):
        Ns, Ni, Na = axes.shape[0], self.Ni, self.Na
        [(axes[s].set_adjustable('box'), axes[s].set_aspect('equal')) for s in range(Ns)]
        [(axes[s].set_xlim(0, 1), axes[s].set_ylim(0, 1)) for s in range(Ns)]
        [[axes[s].plot(*np.array([[0, 1], [0, 0]])[[i, 1-i], :], 'k', clip_on=False) for i in range(Ni)] for s in range(Ns)]
        [[axes[s].plot(*np.array([1, 0])[[i, 1-i], np.newaxis], f'k{[">", "^"][i]}', clip_on=False) for i in range(Ni)] for s in range(Ns)]
        [[axes[s].annotate(rf'$\pi_0^{{{s}{i}}}$', np.array([1, 0])[[i, 1-i]], xytext=1.2*np.array([2*i-1, 1-2*i]), textcoords='offset fontsize', fontweight='bold', ha='center', va='center') for i in range(Ni)] for s in range(Ns)]
        if plot_traj:
            [axes[s].plot(*self.policy_f[:, s, :, 0].swapaxes(0, 1), 'C7.', markersize=1.0) for s in range(Ns)]
            del self.policy_f
        hypesurf_plot = [[axes[s].plot([], [], 'g', label=r'$\hat{\pi}_a^i=M(\mu_a^i)(\pi_a^{i-})$')[0] for i in range(Ni)] for s in range(Ns)]
        rect_plot = [axes[s].plot([], [], 'b:', linewidth=1.0)[0] for s in range(Ns)]
        policy_bias_plot = [axes[s].plot([], [], 'y^-', label=r'$\pi_a^i-\hat{\pi}_a^i$')[0] for s in range(Ns)]
        tanoffset_plot = [axes[s].plot([], [], 'k.', markersize=4.0, zorder=12)[0] for s in range(Ns)]
        tanvec_plot = [[[axes[s].quiver([], [], [], [], scale=1, width=0.004, color='r', zorder=10) for a in range(Na)] for i in range(Ni)] for s in range(Ns)]
        tanvec_legend = [axes[s].plot([], [], color='r', linestyle='', marker=r'$\longrightarrow$', markersize=15, label=r"$\mu_{a''}^k d\pi_{a'}^j/d\mu_{a''}^k$")[0] for s in range(Ns)]
        fig.legend(handles=[hypesurf_plot[0][0], policy_bias_plot[0], tanvec_legend[0]], loc=legend_loc, ncol=legend_ncol)
        return hypesurf_plot, rect_plot, policy_bias_plot, tanoffset_plot, tanvec_plot

    def kktcondition_update(self, k, data, plots):
        prange, hypesurf, rect, policy_bias, unbiased_policy, unbpol_invindex, tanvec = data
        hypesurf_plot, rect_plot, policy_bias_plot, tanoffset_plot, tanvec_plot = plots
        Ns, Ni, Na = len(hypesurf_plot), self.Ni, self.Na
        [[hypesurf_plot[s][i].set_data(*np.array([hypesurf[i, k, s, :], prange])[[i, 1-i]]) for i in range(Ni)] for s in range(Ns)]
        [rect_plot[s].set_data(*rect[:, k, s, :]) for s in range(Ns)]
        [policy_bias_plot[s].set_data(*policy_bias[:, k, s, :]) for s in range(Ns)]
        [tanoffset_plot[s].set_data(unbiased_policy[unbpol_invindex[k], s, :, 0, np.newaxis]) for s in range(Ns)]
        [[[tanvec_plot[s][i][a].set_offsets(unbiased_policy[unbpol_invindex[k], s, :, 0]) for a in range(Na)] for i in range(Ni)] for s in range(Ns)]
        [[[tanvec_plot[s][i][a].set_UVC(*tanvec[unbpol_invindex[k], s, :, 0, i, a]) for a in range(Na)] for i in range(Ni)] for s in range(Ns)]

    def _curve_plot(self, ax, data, n, label):
        color = ["g", "r"][n]
        plot_handle = ax.plot(np.arange(self.num), data, color, label=label)[0]
        ax.set_ylabel(label, fontweight='bold')
        ax.yaxis.label.set_color(color)
        ax.tick_params(axis='y', colors=color)
        return plot_handle

    def curve_data(self, barr, policy, regret, value):
        norm = np.linalg.norm
        upi, I_gammaTpi, res, upi_, I_gammaTpi_, resb = self._get_cone_data(policy, value)
        res_norm, res_direc = norm(res, np.inf, axis=1).max(axis=1), (self.Ns*norm(res, axis=1)**2/np.sum(res, axis=1)**2-1).clip(min=0).max(axis=1)**0.5
        dual_policy, dual_regret = barr/regret, barr/policy
        policy_bias, regret_bias = policy-dual_policy, regret-dual_regret
        policy_bias_norm, regret_bias_norm = norm(policy_bias, np.inf, axis=-1).max(axis=(-1, -2)), norm(regret_bias, np.inf, axis=-1).max(axis=(-1, -2))
        barr_norm, cano_sect = np.log(barr).max(axis=(1, 2, 3)), norm(policy*((lambda piU_vec: piU_vec.max(axis=-1)[..., np.newaxis]-piU_vec)(value[..., np.newaxis]-resb)), np.inf, axis=-1).max(axis=(1, 2))
        data = [[res_norm, res_direc], [policy_bias_norm, regret_bias_norm], [barr_norm, cano_sect]]
        return data

    def curve_plot(self, fig, item=[0, 1, 2]):
        label = [[r'$V_s^i-D_\pi(V_s^i)$', r'$\tan\measuredangle (V_s^i$''\n'r'$-D_\pi(V_s^i),\mathbf{1}_s^i)$'],
                 [r'$\pi_a^{si}-\hat{\pi}_a^{si}$', r'$r_a^{si}-\hat{r}_a^{si}$'],
                 [r'$\ln\mu_a^{si}$', r'$\bar{\mu}_a^{si}$']]
        axes = fig.subplots(len(item), 1) if len(item) > 1 else [fig.subplots(len(item), 1)]
        [(axes[m].sharex(axes[-1]), axes[m].tick_params('x', labelbottom=False)) for m in range(len(item)-1)]
        axes[-1].set(xlabel='Iterations')
        plots = [[self._curve_plot(ax, self.curve_plot_data[item[m]][n], n, label[item[m]][n]) for n, ax in enumerate([axes[m], axes[m].twinx()])] for m in range(len(item))]
        return axes

    def anim(self):
        def _anim(k):
            self.cone_update(k, *cone_data_plots)
            self.barrproblem_update(k, *barrproblem_data_plots)
            self.kktcondition_update(k, *kktcondition_data_plots)
            [index_plot[m].set_data([frames[k], frames[k]], curvelim[m]) for m in range(3)]
            print(f'{k}/{self.frame_num}')
        barr, policy, regret, value = self.barr, self.policy, self.regret, self.value
        ustatic = self.ua[np.newaxis, ...]+self.gamma*np.moveaxis(np.dot(self.Ta, value), -2, 0)
        frames, frame_num = self.frames, self.frame_num
        fig = plt.figure(figsize=(16, 8), layout='compressed')
        graphfig, curvefig = fig.subfigures(1, 2, width_ratios=[1, 0.5])
        titlefig_, plotfig = graphfig.subfigures(2, 1, height_ratios=[1, 25])
        titlefig = titlefig_.subfigures(1, 3, width_ratios=[0.6, 1, 0.5])
        axes = plotfig.subplots(2, 4)
        [fig.suptitle(title, y=0, va='bottom') for fig, title in zip(titlefig, ['Policy cone', 'Unbiased barriar problem', 'Unbiased KKT conditions'])]
        curveaxes = self.curve_plot(curvefig, item=[0, 1, 2])
        index_plot = [ax.plot([], [], 'k', linewidth=2.0)[0] for ax in curveaxes]
        curvelim = [ax.get_ylim() for ax in curveaxes]
        cone_data_plots = self.cone_data(policy, value), self.cone_plots(plotfig, axes[:, 0], 'outside lower left', 3)
        barrproblem_data_plots = self.barrproblem_data(barr, policy, regret), self.barrproblem_plots(plotfig, axes[:, [1, 2]], 'outside lower center', 3)
        kktcondition_data_plots = self.kktcondition_data(ustatic, barr, policy), self.kktcondition_plots(plotfig, axes[:, 3], 'outside lower right', 1)
        plt.savefig('fig/x.png')
        ani = animation.FuncAnimation(fig, _anim, init_func=lambda: None, repeat=True, frames=frame_num)
        ani.save(f'fig/anim{self.nNn}.gif', fps=60, dpi=200, writer='ffmpeg')

    def graph(self):
        path_, format_ = 'fig', 'eps'
        Ns, Ni, Na = self.Ns, self.Ni, self.Na

        fig, axes = plt.subplots(1, 2, layout='compressed', figsize=(6.4, 3.7))
        np.random.seed(25)
        policy_ = np.random.dirichlet(np.ones(Na), size=(Ns, Ni))
        value_ = np.random.rand(*(Ns, Ni))+np.array([np.ones(Ns)*0.3, np.ones(Ns)*1]).swapaxes(0, 1)
        cone_data_plots = self.cone_data(policy_[np.newaxis, ...], value_[np.newaxis, ...]), self.cone_plots(fig, axes, 'outside lower center', 6, plot_traj=False)
        self.cone_update(0, *cone_data_plots)
        plt.savefig(f'{path_}/cone.{format_}', dpi=300, format=format_)

        fig, axes = plt.subplots(1, 2, layout='compressed', figsize=(6.4, 3.7))
        np.random.seed(0)
        policy_ = np.random.dirichlet(np.ones(Na), size=(Ns, Ni))
        dual_policy_ = np.random.dirichlet(np.ones(Na), size=(Ns, Ni))
        dual_policy_[:, 0, :] = np.random.rand(*(Ns, Na))
        regret_ = np.random.rand(*(Ns, Ni, Na))
        barrproblem_data_plots = self.barrproblem_data((dual_policy_*regret_)[np.newaxis, [0], ...], policy_[np.newaxis, [0], ...], regret_[np.newaxis, [0], ...]), self.barrproblem_plots(fig, axes[np.newaxis, :], 'outside lower center', 4, plot_traj=False)
        self.barrproblem_update(0, *barrproblem_data_plots)
        plt.savefig(f'{path_}/barrproblem.{format_}', dpi=300, format=format_)

        fig, axes = plt.subplots(1, 2, layout='compressed', figsize=(6.4, 3.7))
        ustatic_ = np.array([[[[1e-1, 0]], [[1, 9]]], [[[0, 9]], [[0, 0]]]])
        barr_ = np.array([[[0.05, 0.37], [0.45, 0.74]], [[0.17, 0.5], [0.22, 0.73]]])
        policy_ = (lambda p11, p12, p21, p22: np.array([[[p11, 1-p11], [p12, 1-p12]], [[p21, 1-p21], [p22, 1-p22]]]))(0.3, 0.3, 0.2, 0.3)
        insec_policy_ = (lambda p11, p12, p21, p22: np.array([[[p11, 1-p11], [p12, 1-p12]], [[p21, 1-p21], [p22, 1-p22]]]))(0.605, 0.162, 0.396, 0.669)
        kktcondition_data_plots = self.kktcondition_data(ustatic_[np.newaxis, ...][[0, 0]][..., [0, 0], :], barr_[np.newaxis, ...][[0, 0]], np.array([policy_, insec_policy_])), self.kktcondition_plots(fig, axes, 'outside lower center', 4, plot_traj=False)
        self.kktcondition_update(0, *kktcondition_data_plots)
        plt.savefig(f'{path_}/kktcondition.{format_}', dpi=300, format=format_)

        self.curve_plot(plt.figure(layout='compressed', figsize=(8, 4)))
        plt.savefig(f'{path_}/iter_curve.{format_}', dpi=300, format=format_)
