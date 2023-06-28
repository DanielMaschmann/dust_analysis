"""
script to test and devlop xgalspec tool
"""

from xgalspectool.rcsed_spec_fit import RCSEDSpecFit
from xgalspectool import fit_plot
import numpy as np
import matplotlib.pyplot as plt
import xgalimgtool
from xgalimgtool.imgtool import ImgTool
import dust_tools


mjd_mrk33 = 52643
plate_mrk33 = 905
fiberid_mrk33 = 497
ra_mrk33 = 158.132820
dec_mrk33 = 54.401044


rcsed_object = RCSEDSpecFit(sdss_mjd=mjd_mrk33, sdss_plate=plate_mrk33, sdss_fiberid=fiberid_mrk33, rcsed_version=2)

# if line list is None we use a very standard line list
ln_list = [4342, 4863]

fit_result_dict = rcsed_object.run_rcsed_em_fit(n_nl_gauss=1, n_bl_gauss=0, n_nl_lorentz=0, ln_list=ln_list)
print(fit_result_dict)

# claculate E(B-V)
dust_class = dust_tools.extinction_tools.ExtinctionTools()
ebv_balmer = dust_class.get_balmer_extinct_beta_gamma(flux_h_beta_4863=fit_result_dict['nl_gauss_0']['flux_4863'],
                                                      flux_h_gamma_4342=fit_result_dict['nl_gauss_0']['flux_4342'])
print('ebv_balmer ', ebv_balmer)


# get the instrumental broadening for each line
inst_broad_dict = {}
for line in ln_list:
    inst_broad_dict.update(
        {line: rcsed_object.get_rcsed_line_inst_broad(line=line, redshift=rcsed_object.get_rcsed_spec_redshift())})

# getting the sprectral data
wave, em_flux, em_flux_err = rcsed_object.get_rcsed_em_spec()
em_flux_err = np.ones(len(em_flux))
# mask only the parts of the spectrum where we have a line we want to fit
line_mask = rcsed_object.get_rcsed_multiple_line_masks(line_list=ln_list, redshift=rcsed_object.get_rcsed_spec_redshift())

# get systematic velocity
sys_vel = rcsed_object.get_rcsed_sys_vel(redshift=rcsed_object.get_rcsed_spec_redshift())

image_access = ImgTool(ra=ra_mrk33, dec=dec_mrk33)
g_header, img_rgb = image_access.get_legacy_survey_coord_img(height=120, width=120)


plotting_class = fit_plot.FitPlot(x_data=wave, flux=em_flux, flux_err=em_flux_err, sys_vel=sys_vel,
                 inst_broad_dict=inst_broad_dict, fit_result_dict=fit_result_dict,
                 n_nl_gauss=1, n_nl_lorentz=0, n_bl_gauss=0)


fig = plotting_class.plot_single_gauss_fit_results_simple_beta_gamma(header=g_header, img=img_rgb, ra=ra_mrk33, dec=dec_mrk33, fiber_radius=1.5)

fig.savefig('plot_output/em_fit.png')

balmer_dict = {
    'wavelength': wave,
    'total_flux': em_flux,
    'h_beta_flux': fit_result_dict['nl_gauss_0']['flux_4863'],
    'h_gamma_flux': fit_result_dict['nl_gauss_0']['flux_4342'],
    'h_beta_flux_err': fit_result_dict['nl_gauss_0']['flux_4863_err'],
    'h_gamma_flux_err': fit_result_dict['nl_gauss_0']['flux_4342_err'],

}

np.save('data_output/balmer_dict_MRK_33A.npy', balmer_dict)

exit()

ebv_balmer = dust_class.get_balmer_extinct(flux_h_alpha_6565=fit_result_dict['nl_gauss_0']['flux_6565'],
                                           flux_h_beta_4863=fit_result_dict['nl_gauss_0']['flux_4863'])


exit()



exit()

plotting_class = fit_plot.FitPlot(x_data=wave, flux=em_flux, flux_err=em_flux_err, sys_vel=sys_vel,
                 inst_broad_dict=inst_broad_dict, fit_result_dict=fit_result_dict,
                 n_nl_gauss=1, n_nl_lorentz=0, n_bl_gauss=0)


fig = plotting_class.plot_single_gauss_fit_results_simple(header=g_header, img=img_rgb, ra=ra_mrk33, dec=dec_mrk33, fiber_radius=1.5)

fig.savefig('plot_output/em_fit.png')

# claculate E(B-V)
dust_class = dust_tools.extinction_tools.ExtinctionTools()

ebv_balmer = dust_class.get_balmer_extinct(flux_h_alpha_6565=fit_result_dict['nl_gauss_0']['flux_6565'],
                                           flux_h_beta_4863=fit_result_dict['nl_gauss_0']['flux_4863'])

print('ebv_balmer ', ebv_balmer)
