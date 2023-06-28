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


mjd_ngc4670 = 54205
plate_ngc4670 = 2238
fiberid_ngc4670 = 222
ra_ngc4670 = 191.321910
dec_ngc4670 = 27.125597


rcsed_object = RCSEDSpecFit(sdss_mjd=mjd_ngc4670, sdss_plate=plate_ngc4670, sdss_fiberid=fiberid_ngc4670, rcsed_version=2)


fit_result_dict = rcsed_object.run_rcsed_em_fit(n_nl_gauss=1, n_bl_gauss=0, n_nl_lorentz=0)


# if line list is None we use a very standard line list
ln_list = [4863, 5008, 6550, 6565, 6585]

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

image_access = ImgTool(ra=ra_ngc4670, dec=dec_ngc4670)
g_header, img_rgb = image_access.get_legacy_survey_coord_img(height=120, width=120)


print(fit_result_dict)
plotting_class = fit_plot.FitPlot(x_data=wave, flux=em_flux, flux_err=em_flux_err, sys_vel=sys_vel,
                 inst_broad_dict=inst_broad_dict, fit_result_dict=fit_result_dict,
                 n_nl_gauss=1, n_nl_lorentz=0, n_bl_gauss=0)


fig = plotting_class.plot_single_gauss_fit_results_simple(header=g_header, img=img_rgb, ra=ra_ngc4670, dec=dec_ngc4670, fiber_radius=1.5)

fig.savefig('plot_output/em_fit.png')

# claculate E(B-V)
dust_class = dust_tools.extinction_tools.ExtinctionTools()

balmer_dict = {
    'wavelength': wave,
    'total_flux': em_flux,
    'h_alpha_flux': fit_result_dict['nl_gauss_0']['flux_6565'],
    'h_beta_flux': fit_result_dict['nl_gauss_0']['flux_4863'],
    'h_alpha_flux_err': fit_result_dict['nl_gauss_0']['flux_6565_err'],
    'h_beta_flux_err': fit_result_dict['nl_gauss_0']['flux_4863_err'],
}

np.save('data_output/balmer_dict_NGC_4670.npy', balmer_dict)




# ebv_balmer = dust_class.get_balmer_extinct(flux_h_alpha_6565=fit_result_dict['nl_gauss_0']['flux_6565'],
#                                            flux_h_beta_4863=fit_result_dict['nl_gauss_0']['flux_4863'])
#
# print('ebv_balmer ', ebv_balmer)
