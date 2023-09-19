import numpy as np

import matplotlib.pyplot as plt

calzetti_windows = np.array([(1265, 1285),
                             (1305, 1320),
                             (1345, 1370),
                             (1415, 1490),
                             (1565, 1585),
                             (1625, 1630),
                             (1658, 1673),
                             (1695, 1700)])

plotting_params = {
    'HE_2_10A': {
        'ylim_g140l': (-0.25 * 1e-14, 1.8 * 1e-14)},
    'HE_2_10B': {
        'ylim_g140l': (-0.25 * 1e-14, 1.5 * 1e-14)},
    'HE_2_10C': {
        'ylim_g140l': (-0.25 * 1e-14, 1.05 * 1e-14)},
    'HE_2_10D': {
        'ylim_g140l': (-0.25 * 1e-14, 0.8 * 1e-14)},
    'MRK_33A': {
        'ylim_g140l': (-0.15 * 1e-14, 0.7 * 1e-14)},
    'MRK_33B': {
        'ylim_g140l': (-0.15 * 1e-14, 0.7 * 1e-14)},
    'NGC_3049A': {
        'ylim_g140l': (-0.15 * 1e-14, 2.4 * 1e-14)},
    'NGC_3049B': {
        'ylim_g140l': (-0.15 * 1e-14, 0.3 * 1e-14)},
    'NGC_3125A': {
        'ylim_g140l': (-0.25 * 1e-14, 1.8 * 1e-14)},
    'NGC_4214A': {
        'ylim_g140l': (-0.25 * 1e-14, 9.8 * 1e-14)},
    'NGC_4670A': {
        'ylim_g140l': (-0.25 * 1e-14, 1.8 * 1e-14)},
    'NGC_4670B': {
        'ylim_g140l': (-0.25 * 1e-14, 0.7 * 1e-14)},
    'NGC_4670C': {
        'ylim_g140l': (-0.25 * 1e-14, 0.7 * 1e-14)},
    'TOL_1924_416A': {
        'ylim_g140l': (-0.25 * 1e-14, 3.8 * 1e-14)},
    'TOL_1924_416B': {
        'ylim_g140l': (-0.15 * 1e-14, 0.7 * 1e-14)},
    'TOL_89A': {
        'ylim_g140l': (-0.25 * 1e-14, 1.8 * 1e-14)}
}


def plot_g140l_line(target_id, ax_g140l, y_offset=0.0, name_track=False, name_offset=0.0):
    fit_dict = np.load('data_output/fit_dict_%s.npy' % target_id, allow_pickle=True).item()

    # get data
    continuum_g140l = fit_dict['continuum_g140l']
    wave_g140l = fit_dict['wave_g140l']
    flux_g140l = fit_dict['flux_g140l']
    flux_err_g140l = fit_dict['flux_err_g140l']
    continuum_g430m = fit_dict['continuum_g430m']
    wave_g430m = fit_dict['wave_g430m']
    flux_g430m = fit_dict['flux_g430m']
    flux_err_g430m = fit_dict['flux_err_g430m']
    beta = fit_dict['beta']
    beta_err = fit_dict['beta_err']
    line_flux_f1640 = fit_dict['line_flux_f1640']
    line_flux_err_f1640 = fit_dict['line_flux_err_f1640']
    line_flux_f4686 = fit_dict['line_flux_f4686']
    line_flux_err_f4686 = fit_dict['line_flux_err_f4686']


    # plot g140l
    ax_g140l.step(wave_g140l, flux_g140l + y_offset, where='mid', color='k', linewidth=2)
    for window in calzetti_windows:
        mask_include_fit = (wave_g140l > window[0]) & (wave_g140l < window[1])
        ax_g140l.step(wave_g140l[mask_include_fit], flux_g140l[mask_include_fit] + y_offset, where='mid', color='tab:red',
                      linewidth=2)
    ax_g140l.fill_between(wave_g140l, (flux_g140l - flux_err_g140l) + y_offset, (flux_g140l + flux_err_g140l) + y_offset,
                          color='gray', alpha=0.7)
    ax_g140l.plot([], [], color='white')
    ax_g140l.plot(wave_g140l, continuum_g140l + y_offset, linestyle='--', linewidth=2, color='tab:blue')
    in_window = (wave_g140l > 1630) & (wave_g140l < 1652)
    ax_g140l.step(wave_g140l[in_window], flux_g140l[in_window] + y_offset, c='cyan', lw=2)


    if name_track:
        xlim_f1640 = (1430, 1640 + 50)

        mask_wave_f1640 = (wave_g140l > xlim_f1640[0]) & (wave_g140l < xlim_f1640[1])
        if y_offset == 0:
            string_name = target_id[-1].upper()
        else:
            string_name = target_id[-1].upper() + ' + %.1f' % (y_offset * 1e14)

        ax_g140l.text(wave_g140l[mask_wave_f1640][0],
                continuum_g140l[mask_wave_f1640][0] + y_offset + name_offset,
                string_name,
                horizontalalignment='left', verticalalignment='bottom',
                fontsize=fontsize - 5)



def plot_ax_f4686_line(target_id, ax_f4686, y_offset=0.0, name_track=False, name_offset=0.0):
    fit_dict = np.load('data_output/fit_dict_%s.npy' % target_id, allow_pickle=True).item()

    # get data
    continuum_g140l = fit_dict['continuum_g140l']
    wave_g140l = fit_dict['wave_g140l']
    flux_g140l = fit_dict['flux_g140l']
    flux_err_g140l = fit_dict['flux_err_g140l']
    continuum_g430m = fit_dict['continuum_g430m']
    wave_g430m = fit_dict['wave_g430m']
    flux_g430m = fit_dict['flux_g430m']
    flux_err_g430m = fit_dict['flux_err_g430m']
    beta = fit_dict['beta']
    beta_err = fit_dict['beta_err']
    line_flux_f1640 = fit_dict['line_flux_f1640']
    line_flux_err_f1640 = fit_dict['line_flux_err_f1640']
    line_flux_f4686 = fit_dict['line_flux_f4686']
    line_flux_err_f4686 = fit_dict['line_flux_err_f4686']
    # plot g140l
    ax_f4686.step(wave_g430m, flux_g430m + y_offset, where='mid', color='k', linewidth=2)

    ax_f4686.fill_between(wave_g430m, (flux_g430m-flux_err_g430m) + y_offset, (flux_g430m+flux_err_g430m) + y_offset,
                          color='gray', alpha=0.7)
    ax_f4686.plot(wave_g430m, continuum_g430m + y_offset, linestyle='--', linewidth=2, color='tab:blue')
    if target_id == 'MRK_33B':
        # in_window = (wave_g430m > 4670) & (wave_g430m < 4705)
        in_window = ((wave_g430m >= 4670) & (wave_g430m <= 4676) |
                     (wave_g430m >= 4678.5) & (wave_g430m <= 4705))

    else:
        in_window = (wave_g430m > 4670) & (wave_g430m < 4705)
    ax_f4686.step(wave_g430m[in_window], flux_g430m[in_window] + y_offset, c='cyan', lw=2)

    if name_track:
        xlim_f4686 = (4730, 4730 + 50)

        mask_wave_f4686 = (wave_g430m > xlim_f4686[0]) & (wave_g430m < xlim_f4686[1])
        if y_offset == 0:
            string_name = target_id[-1].upper()
        else:
            string_name = target_id[-1].upper() + ' + %.1f' % (y_offset * 1e15)

        ax_f4686.text(wave_g430m[mask_wave_f4686][0],
                continuum_g430m[mask_wave_f4686][0] + y_offset + name_offset,
                string_name,
                horizontalalignment='left', verticalalignment='bottom',
                fontsize=fontsize - 5)


def plot_line_fluxes_f4686(target_id_list, ax, fontsize):
    for target_id in target_id_list:

        fit_dict = np.load('data_output/fit_dict_%s.npy' % target_id, allow_pickle=True).item()

        # get data
        line_flux_f4686 = fit_dict['line_flux_f4686']
        line_flux_err_f4686 = fit_dict['line_flux_err_f4686']

        ax.scatter([],[], color='white', s=0.0001, label=
        r'HeII(4686)$_{\rm %s}$ = %.2f $^{+%.2f}_{-%.2f} \times 10^{-15}$ erg/s/cm$^{2}$ ' % \
                (target_id[-1], line_flux_f4686, line_flux_err_f4686[0], line_flux_err_f4686[1]))
    ax.legend(frameon=False, fontsize=fontsize-10, loc='upper left', bbox_to_anchor=[-0.1, 0.98], )




def plot_line_fluxes_g140l(target_id_list, ax, fontsize):
    for target_id in target_id_list:

        fit_dict = np.load('data_output/fit_dict_%s.npy' % target_id, allow_pickle=True).item()

        # get data
        continuum_g140l = fit_dict['continuum_g140l']
        wave_g140l = fit_dict['wave_g140l']
        flux_g140l = fit_dict['flux_g140l']
        flux_err_g140l = fit_dict['flux_err_g140l']
        continuum_g430m = fit_dict['continuum_g430m']
        wave_g430m = fit_dict['wave_g430m']
        flux_g430m = fit_dict['flux_g430m']
        flux_err_g430m = fit_dict['flux_err_g430m']
        beta = fit_dict['beta']
        beta_err = fit_dict['beta_err']
        line_flux_f1640 = fit_dict['line_flux_f1640']
        line_flux_err_f1640 = fit_dict['line_flux_err_f1640']
        line_flux_f4686 = fit_dict['line_flux_f4686']
        line_flux_err_f4686 = fit_dict['line_flux_err_f4686']

        ax.scatter([],[], color='white', s=0.00001, label=
        r'$\beta_{\rm %s} = %.2f \pm %.2f$, HeII(1640)$_{\rm %s}$ = %.2f $^{+%.2f}_{-%.2f} \times 10^{-15}$ erg/s/cm$^{2}$ ' % \
                (target_id[-1], beta, beta_err, target_id[-1], line_flux_f1640, line_flux_err_f1640[0], line_flux_err_f1640[1]))

    ax.legend(frameon=False, fontsize=fontsize-10, loc='upper center')



def adjust_axis(ax, xlim=(1150, 1699), ylim=(-0.25 * 1e-14, 1.8 * 1e-14), ylabelpad=0, xlabelpad=0,
                hatch_lim=(4670, 4705), filter_name='G140L', filter_name_offset=0.92,
                target_name=None, target_name_offset=0.02,
                 fontsize=25, x_ticks=True, x_label=True, y_ticks=True, y_label=True):

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.tick_params(axis='both', which='both', width=3, length=6, right=True, top=True, direction='in',
                         labelsize=fontsize)
    ax.yaxis.offsetText.set_fontsize(fontsize - 10)

    if not x_ticks:
        ax.set_xticklabels([])

    if x_label:
        ax.set_xlabel('Rest Wavelength (Å)', labelpad=xlabelpad, fontsize=fontsize)

    if not y_ticks:
        ax.set_yticklabels([])

    if y_label:
        ax.set_ylabel('Flux (erg/s/cm/cm/Å)', labelpad=ylabelpad, fontsize=fontsize)

    ax.fill_between([hatch_lim[0], hatch_lim[1]], ax.get_ylim()[1], ax.get_ylim()[0], color="none",
                          edgecolor='gray', hatch="////")


    ax.text(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * filter_name_offset,
            ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.95,
            filter_name,
            horizontalalignment='right', verticalalignment='top',
            fontsize=fontsize - 4)
    if target_name is not None:
        ax.text(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * target_name_offset,
                ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.95,
                target_name,
                horizontalalignment='left', verticalalignment='top',
                fontsize=fontsize)



# create plotting
figure = plt.figure(figsize=(30, 40))
fontsize = 35
# ax_g140l_he210 = figure.add_axes([0.035, 0.82, 0.605, 0.17])
# ax_f4686_he210 = figure.add_axes([0.67, 0.82, 0.325, 0.17])
#
# ax_g140l_mrk33 = figure.add_axes([0.035, 0.69, 0.605, 0.12])
# ax_f4686_mrk33 = figure.add_axes([0.67, 0.69, 0.325, 0.12])
#
# ax_g140l_ngc3049 = figure.add_axes([0.035, 0.56, 0.605, 0.12])
# ax_f4686_ngc3049 = figure.add_axes([0.67, 0.56, 0.325, 0.12])
#
# ax_g140l_ngc3125 = figure.add_axes([0.035, 0.46, 0.605, 0.09])
# ax_f4686_ngc3125 = figure.add_axes([0.67, 0.46, 0.325, 0.09])
#
# ax_g140l_ngc4214 = figure.add_axes([0.035, 0.36, 0.605, 0.09])
# ax_f4686_ngc4214 = figure.add_axes([0.67, 0.36, 0.325, 0.09])
#
# ax_g140l_ngc4670 = figure.add_axes([0.035, 0.23, 0.605, 0.12])
# ax_f4686_ngc4670 = figure.add_axes([0.67, 0.23, 0.325, 0.12])
#
# ax_g140l_tol89 = figure.add_axes([0.035, 0.13, 0.605, 0.09])
# ax_f4686_tol89 = figure.add_axes([0.67, 0.13, 0.325, 0.09])
#
# ax_g140l_tol1924_416 = figure.add_axes([0.035, 0.03, 0.605, 0.09])
# ax_f4686_tol1924_416 = figure.add_axes([0.67, 0.03, 0.325, 0.09])


ax_g140l_he210 = figure.add_axes([0.035, 0.82, 0.605, 0.17])
ax_f4686_he210 = figure.add_axes([0.67, 0.82, 0.325, 0.17])

ax_g140l_ngc3049 = figure.add_axes([0.035, 0.69, 0.605, 0.12])
ax_f4686_ngc3049 = figure.add_axes([0.67, 0.69, 0.325, 0.12])

ax_g140l_ngc3125 = figure.add_axes([0.035, 0.59, 0.605, 0.09])
ax_f4686_ngc3125 = figure.add_axes([0.67, 0.59, 0.325, 0.09])


ax_g140l_mrk33 = figure.add_axes([0.035, 0.46, 0.605, 0.12])
ax_f4686_mrk33 = figure.add_axes([0.67, 0.46, 0.325, 0.12])


ax_g140l_ngc4214 = figure.add_axes([0.035, 0.36, 0.605, 0.09])
ax_f4686_ngc4214 = figure.add_axes([0.67, 0.36, 0.325, 0.09])

ax_g140l_ngc4670 = figure.add_axes([0.035, 0.23, 0.605, 0.12])
ax_f4686_ngc4670 = figure.add_axes([0.67, 0.23, 0.325, 0.12])


ax_g140l_tol1924_416 = figure.add_axes([0.035, 0.13, 0.605, 0.09])
ax_f4686_tol1924_416 = figure.add_axes([0.67, 0.13, 0.325, 0.09])

ax_g140l_tol89 = figure.add_axes([0.035, 0.03, 0.605, 0.09])
ax_f4686_tol89 = figure.add_axes([0.67, 0.03, 0.325, 0.09])


plot_g140l_line(target_id='HE_2_10A', ax_g140l=ax_g140l_he210, y_offset=0.8 * 1e-14, name_track=True, name_offset=0)
plot_g140l_line(target_id='HE_2_10B', ax_g140l=ax_g140l_he210, y_offset=0.6 * 1e-14, name_track=True, name_offset=0)
plot_g140l_line(target_id='HE_2_10C', ax_g140l=ax_g140l_he210, y_offset=0.2 * 1e-14, name_track=True, name_offset=0)
plot_g140l_line(target_id='HE_2_10D', ax_g140l=ax_g140l_he210, y_offset=0.0 * 1e-14, name_track=True, name_offset=0)
plot_ax_f4686_line(target_id='HE_2_10A', ax_f4686=ax_f4686_he210, y_offset=0.8 * 1e-15, name_track=True, name_offset=0.2 * 1e-15)
plot_ax_f4686_line(target_id='HE_2_10B', ax_f4686=ax_f4686_he210, y_offset=1.0 * 1e-15, name_track=True, name_offset=0.2 * 1e-15)
plot_ax_f4686_line(target_id='HE_2_10C', ax_f4686=ax_f4686_he210, y_offset=0.2 * 1e-15, name_track=True, name_offset=0.2 * 1e-15)
plot_ax_f4686_line(target_id='HE_2_10D', ax_f4686=ax_f4686_he210, y_offset=0.0 * 1e-15, name_track=True, name_offset=0.2 * 1e-15)

plot_g140l_line(target_id='MRK_33A', ax_g140l=ax_g140l_mrk33, y_offset=0.20 * 1e-14, name_track=True, name_offset=0.02 * 1e-14)
plot_g140l_line(target_id='MRK_33B', ax_g140l=ax_g140l_mrk33, y_offset=0.0 * 1e-14, name_track=True, name_offset=0.02 * 1e-14)
plot_ax_f4686_line(target_id='MRK_33A', ax_f4686=ax_f4686_mrk33, y_offset=0.4 * 1e-15, name_track=True, name_offset=0.1 * 1e-15)
plot_ax_f4686_line(target_id='MRK_33B', ax_f4686=ax_f4686_mrk33, y_offset=0.0 * 1e-15, name_track=True, name_offset=0.1 * 1e-15)

plot_g140l_line(target_id='NGC_3049A', ax_g140l=ax_g140l_ngc3049, y_offset=0.0 * 1e-14, name_track=True, name_offset=0.02 * 1e-14)
plot_g140l_line(target_id='NGC_3049B', ax_g140l=ax_g140l_ngc3049, y_offset=0.0 * 1e-14, name_track=True, name_offset=0.02 * 1e-14)
plot_ax_f4686_line(target_id='NGC_3049A', ax_f4686=ax_f4686_ngc3049, y_offset=0.0 * 1e-15, name_track=True, name_offset=0.1 * 1e-15)
plot_ax_f4686_line(target_id='NGC_3049B', ax_f4686=ax_f4686_ngc3049, y_offset=0.0 * 1e-15, name_track=True, name_offset=0.1 * 1e-15)


plot_g140l_line(target_id='NGC_3125A', ax_g140l=ax_g140l_ngc3125, y_offset=0.0 * 1e-14, name_track=True, name_offset=0.02 * 1e-14)
plot_ax_f4686_line(target_id='NGC_3125A', ax_f4686=ax_f4686_ngc3125, y_offset=0.0 * 1e-15, name_track=True, name_offset=0.1 * 1e-15)

plot_g140l_line(target_id='NGC_4214A', ax_g140l=ax_g140l_ngc4214, y_offset=0.0 * 1e-14, name_track=True, name_offset=0.02 * 1e-14)
plot_ax_f4686_line(target_id='NGC_4214A', ax_f4686=ax_f4686_ngc4214, y_offset=0.0 * 1e-15, name_track=True, name_offset=0.1 * 1e-15)

plot_g140l_line(target_id='NGC_4670A', ax_g140l=ax_g140l_ngc4670, y_offset=0.3 * 1e-14, name_track=True, name_offset=0.02 * 1e-14)
plot_g140l_line(target_id='NGC_4670B', ax_g140l=ax_g140l_ngc4670, y_offset=0.3 * 1e-14, name_track=True, name_offset=0.02 * 1e-14)
plot_g140l_line(target_id='NGC_4670C', ax_g140l=ax_g140l_ngc4670, y_offset=0.0 * 1e-14, name_track=True, name_offset=0.02 * 1e-14)
plot_ax_f4686_line(target_id='NGC_4670A', ax_f4686=ax_f4686_ngc4670, y_offset=0.3 * 1e-15, name_track=True, name_offset=0.1 * 1e-15)
plot_ax_f4686_line(target_id='NGC_4670B', ax_f4686=ax_f4686_ngc4670, y_offset=0.4 * 1e-15, name_track=True, name_offset=0.1 * 1e-15)
plot_ax_f4686_line(target_id='NGC_4670C', ax_f4686=ax_f4686_ngc4670, y_offset=0.0 * 1e-15, name_track=True, name_offset=0.1 * 1e-15)

plot_g140l_line(target_id='TOL_89A', ax_g140l=ax_g140l_tol89, y_offset=0.0 * 1e-14, name_track=True, name_offset=0.02 * 1e-14)
plot_ax_f4686_line(target_id='TOL_89A', ax_f4686=ax_f4686_tol89, y_offset=0.0 * 1e-15, name_track=True, name_offset=0.1 * 1e-15)

plot_g140l_line(target_id='TOL_1924_416A', ax_g140l=ax_g140l_tol1924_416, y_offset=0.0 * 1e-14, name_track=True, name_offset=0.02 * 1e-14)
plot_ax_f4686_line(target_id='TOL_1924_416A', ax_f4686=ax_f4686_tol1924_416, y_offset=0.0 * 1e-15, name_track=True, name_offset=0.1 * 1e-15)


adjust_axis(ax=ax_g140l_he210, xlim=(1150, 1699), ylim=(-0.25 * 1e-14, 3.1 * 1e-14),
            hatch_lim=(1630, 1652), filter_name='G140L', filter_name_offset=0.98,
            target_name='He 2-10',
             fontsize=fontsize, x_ticks=False, x_label=False, y_label=True, ylabelpad=3)
adjust_axis(ax=ax_f4686_he210, xlim=(4686 - 120, 4686 + 100), ylim=(0.25 * 1e-15, 5.4 * 1e-15),
            hatch_lim=(4670, 4705), filter_name='G430M', filter_name_offset=0.97,
             fontsize=fontsize, x_ticks=False, x_label=False, y_label=False)

adjust_axis(ax=ax_g140l_mrk33, xlim=(1150, 1699), ylim=(-0.01 * 1e-14, 0.6 * 1e-14),
                        hatch_lim=(1630, 1652), filter_name='G140L', filter_name_offset=0.98,
            target_name='MRK 33',
             fontsize=fontsize, x_ticks=False, x_label=False, y_label=False)
adjust_axis(ax=ax_f4686_mrk33, xlim=(4686 - 120, 4686 + 100), ylim=(-0.15 * 1e-15, 1.5 * 1e-15),
            hatch_lim=(4670, 4705), filter_name='G430M', filter_name_offset=0.97,
             fontsize=fontsize, x_ticks=False, x_label=False, y_label=False)


adjust_axis(ax=ax_g140l_ngc3049, xlim=(1150, 1699), ylim=(-0.1 * 1e-14, 1.8 * 1e-14),
                        hatch_lim=(1630, 1652), filter_name='G140L', filter_name_offset=0.98,
            target_name='NGC 3049',
             fontsize=fontsize, x_ticks=False, x_label=False, y_label=False)
adjust_axis(ax=ax_f4686_ngc3049, xlim=(4686 - 120, 4686 + 100), ylim=(-0.15 * 1e-15, 3.5 * 1e-15),
            hatch_lim=(4670, 4705), filter_name='G430M', filter_name_offset=0.97,
             fontsize=fontsize, x_ticks=False, x_label=False, y_label=False)

adjust_axis(ax=ax_g140l_ngc3125, xlim=(1150, 1699), ylim=(-0.15 * 1e-14, 0.9 * 1e-14),
                        hatch_lim=(1630, 1652), filter_name='G140L', filter_name_offset=0.98,
            target_name='NGC 3125',
             fontsize=fontsize, x_ticks=False, x_label=False, y_label=False)
adjust_axis(ax=ax_f4686_ngc3125, xlim=(4686 - 120, 4686 + 100), ylim=(-0.15 * 1e-15, 1.5 * 1e-15),
            hatch_lim=(4670, 4705), filter_name='G430M', filter_name_offset=0.97,
             fontsize=fontsize, x_ticks=False, x_label=False, y_label=False)

adjust_axis(ax=ax_g140l_ngc4214, xlim=(1150, 1699), ylim=(-0.15 * 1e-14, 5.9 * 1e-14),
                        hatch_lim=(1630, 1652), filter_name='G140L', filter_name_offset=0.98,
            target_name='NGC 4214',
             fontsize=fontsize, x_ticks=False, x_label=False, y_label=False)
adjust_axis(ax=ax_f4686_ngc4214, xlim=(4686 - 120, 4686 + 100), ylim=(1.9 * 1e-15, 3.5 * 1e-15),
            hatch_lim=(4670, 4705), filter_name='G430M', filter_name_offset=0.97,
             fontsize=fontsize, x_ticks=False, x_label=False, y_label=False)

adjust_axis(ax=ax_g140l_ngc4670, xlim=(1150, 1699), ylim=(-0.15 * 1e-14, 2.3 * 1e-14),
                        hatch_lim=(1630, 1652), filter_name='G140L', filter_name_offset=0.98,
            target_name='NGC 4670',
             fontsize=fontsize, x_ticks=False, x_label=False, y_label=False)
adjust_axis(ax=ax_f4686_ngc4670, xlim=(4686 - 120, 4686 + 100), ylim=(-0.2 * 1e-15, 2.7 * 1e-15),
            hatch_lim=(4670, 4705), filter_name='G430M', filter_name_offset=0.97,
             fontsize=fontsize, x_ticks=False, x_label=False, y_label=False)

adjust_axis(ax=ax_g140l_tol89, xlim=(1150, 1699), ylim=(-0.15 * 1e-14, 1.3 * 1e-14),
                        hatch_lim=(1630, 1652), filter_name='G140L', filter_name_offset=0.98,
            target_name='TOL 89',
             fontsize=fontsize, x_ticks=True, x_label=True, y_label=False)
adjust_axis(ax=ax_f4686_tol89, xlim=(4686 - 120, 4686 + 100), ylim=(0.0 * 1e-15, 1.2 * 1e-15),
            hatch_lim=(4670, 4705), filter_name='G430M', filter_name_offset=0.97,
             fontsize=fontsize, x_ticks=True, x_label=True, y_label=False)

adjust_axis(ax=ax_g140l_tol1924_416, xlim=(1150, 1699), ylim=(-0.15 * 1e-14, 3.3 * 1e-14),
                        hatch_lim=(1630, 1652), filter_name='G140L', filter_name_offset=0.98,
            target_name='TOL 1924-416',
             fontsize=fontsize, x_ticks=False, x_label=False, y_label=False)
adjust_axis(ax=ax_f4686_tol1924_416, xlim=(4686 - 120, 4686 + 100), ylim=(0.1 * 1e-15, 1.2 * 1e-15),
            hatch_lim=(4670, 4705), filter_name='G430M', filter_name_offset=0.97,
             fontsize=fontsize, x_ticks=False, x_label=False, y_label=False, xlabelpad=-5)


plot_line_fluxes_g140l(target_id_list=['HE_2_10A', 'HE_2_10B', 'HE_2_10C', 'HE_2_10D'],
                       ax=ax_g140l_he210, fontsize=fontsize)
plot_line_fluxes_f4686(target_id_list=['HE_2_10A', 'HE_2_10B', 'HE_2_10C', 'HE_2_10D'],
                       ax=ax_f4686_he210, fontsize=fontsize)

plot_line_fluxes_g140l(target_id_list=['MRK_33A', 'MRK_33B'],
                       ax=ax_g140l_mrk33, fontsize=fontsize)
plot_line_fluxes_f4686(target_id_list=['MRK_33A', 'MRK_33B'],
                       ax=ax_f4686_mrk33, fontsize=fontsize)

plot_line_fluxes_g140l(target_id_list=['NGC_3049A', 'NGC_3049B'],
                       ax=ax_g140l_ngc3049, fontsize=fontsize)
plot_line_fluxes_f4686(target_id_list=['NGC_3049A', 'NGC_3049B'],
                       ax=ax_f4686_ngc3049, fontsize=fontsize)

plot_line_fluxes_g140l(target_id_list=['NGC_3125A'],
                       ax=ax_g140l_ngc3125, fontsize=fontsize)
plot_line_fluxes_f4686(target_id_list=['NGC_3125A'],
                       ax=ax_f4686_ngc3125, fontsize=fontsize)

plot_line_fluxes_g140l(target_id_list=['NGC_4214A'],
                       ax=ax_g140l_ngc4214, fontsize=fontsize)
plot_line_fluxes_f4686(target_id_list=['NGC_4214A'],
                       ax=ax_f4686_ngc4214, fontsize=fontsize)

plot_line_fluxes_g140l(target_id_list=['NGC_4670A', 'NGC_4670B', 'NGC_4670C'],
                       ax=ax_g140l_ngc4670, fontsize=fontsize)
plot_line_fluxes_f4686(target_id_list=['NGC_4670A', 'NGC_4670B', 'NGC_4670C'],
                       ax=ax_f4686_ngc4670, fontsize=fontsize)

plot_line_fluxes_g140l(target_id_list=['TOL_89A'],
                       ax=ax_g140l_tol89, fontsize=fontsize)
plot_line_fluxes_f4686(target_id_list=['TOL_89A'],
                       ax=ax_f4686_tol89, fontsize=fontsize)

plot_line_fluxes_g140l(target_id_list=['TOL_1924_416A'],
                       ax=ax_g140l_tol1924_416, fontsize=fontsize)
plot_line_fluxes_f4686(target_id_list=['TOL_1924_416A'],
                       ax=ax_f4686_tol1924_416, fontsize=fontsize)


figure.savefig('plot_output/heii_spec.png')
figure.savefig('plot_output/heii_spec.pdf')

exit()
