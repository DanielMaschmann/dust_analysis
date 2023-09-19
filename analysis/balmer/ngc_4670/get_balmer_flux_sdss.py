import dust_tools

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c as speed_of_light
from photometry_tools import helper_func, plotting_tools
import astropy.units as u

from astropy.coordinates import SkyCoord

from astropy.visualization.wcsaxes import SphericalCircle

import spec_fit

import xgalimgtool
from xgalimgtool.imgtool import ImgTool
from astropy.wcs import WCS


mjd_ngc4670 = 54205
plate_ngc4670 = 2238
fiberid_ngc4670 = 222
ra_ngc4670 = 191.321910
dec_ngc4670 = 27.125597


rcsed_object = spec_fit.rcsed_spec_fit.RCSEDSpecFit(sdss_mjd=mjd_ngc4670, sdss_plate=plate_ngc4670,
                                                    sdss_fiberid=fiberid_ngc4670, rcsed_version=1)

rcsed_object.download_rcsed_spec_fits_file()

# ln_list = [4342, 4863]
ln_list = [4863, 5008, 6550, 6565, 6585]

fit_result_dict = rcsed_object.run_rcsed_em_fit(n_nl_gauss=1, n_bl_gauss=0, n_nl_lorentz=0, ln_list=ln_list)

print(fit_result_dict)
#
# plt.errorbar(fit_result_dict['wave'][fit_result_dict['ln_mask']],
#              fit_result_dict['em_flux'][fit_result_dict['ln_mask']],
#              yerr=fit_result_dict['em_flux_err'][fit_result_dict['ln_mask']])
# plt.plot(fit_result_dict['wave'][fit_result_dict['ln_mask']],
#              fit_result_dict['best_fit'])
# plt.show()



# claculate E(B-V)
dust_class = dust_tools.extinction_tools.ExtinctionTools()
ebv_balmer = dust_class.get_balmer_extinct_alpha_beta(flux_h_beta_4863=fit_result_dict['flux_nl_4863_gauss_0'],
                                                      flux_h_alpha_6565=fit_result_dict['flux_nl_6565_gauss_0'])
ebv_balmer_err = dust_class.get_balmer_extinct_alpha_beta_err(flux_h_alpha_6565=fit_result_dict['flux_nl_6565_gauss_0'],
                                                              flux_h_beta_4863=fit_result_dict['flux_nl_4863_gauss_0'],
                                                              flux_h_alpha_6565_err=fit_result_dict['flux_nl_6565_gauss_0_err'],
                                                              flux_h_beta_4863_err=fit_result_dict['flux_nl_4863_gauss_0_err'])

balmer_dict = {
    'h_alpha_flux': fit_result_dict['flux_nl_6565_gauss_0'],
    'h_beta_flux': fit_result_dict['flux_nl_4863_gauss_0'],
    'h_alpha_flux_err': fit_result_dict['flux_nl_6565_gauss_0_err'],
    'h_beta_flux_err': fit_result_dict['flux_nl_4863_gauss_0_err'],
}

np.save('data_output/balmer_dict_NGC_4670.npy', balmer_dict)

