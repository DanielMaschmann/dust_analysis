from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from os import path
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import ppxf.miles_util as lib
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from photometry_tools import helper_func
C = 299792.458  # speed of light in km/s
from astropy.wcs.utils import fit_wcs_from_points
from regions import PixCoord, RectanglePixelRegion
from astroquery.skyview import SkyView
from astroquery.simbad import Simbad

from photometry_tools import plotting_tools

def plot_slit(ax, coords_slit_pix, slit_length, slit_width, slit_pa, plot_scatter=False, color='red', lw=4, linestyle='-'):
    x_cent = float(coords_slit_pix[0])
    y_cent = float(coords_slit_pix[1])

    reg = RectanglePixelRegion(PixCoord(x=x_cent, y=y_cent), width=slit_length,
                               height=slit_width, angle=slit_pa*u.deg)
    reg.plot(ax=ax, edgecolor=color, linewidth=lw, linestyle=linestyle)
    if plot_scatter:
        ax.scatter(coords_slit_pix[0], coords_slit_pix[1], c='r', s=120)


data = np.genfromtxt('data/NGC_4214:S:Opt:mk2006_NED.txt', )

lam = data[:, 0]
galaxy = data[:, 1]
galaxy_noise = data[:, 2]
redshift = 0.0010

lam_range_temp = [3611.4, 6886.6]   # Focus on optical regio
# galaxy = np.sum(cube_muse[:, mask_spectrum], 1)
# lam = wave_muse
w = (lam > lam_range_temp[0]) & (lam < lam_range_temp[1])
galaxy = galaxy[w]
galaxy_noise = galaxy_noise[w]
lam = lam[w]

lam_range_temp = [np.min(lam), np.max(lam)]
velscale = C*np.diff(np.log(lam[-2:]))  # Smallest velocity step
spectra_muse, ln_lam_gal, velscale_muse = util.log_rebin(lam_range_temp, galaxy, velscale=velscale)
spectra_muse_noise, ln_lam_gal_noise, velscale_muse_noise = util.log_rebin(lam_range_temp, galaxy_noise, velscale=velscale)

# print('spectra_muse ', spectra_muse)
# print('ln_lam_gal ', ln_lam_gal)
# print('velscale_muse ', velscale_muse)

velscale = C*np.diff(ln_lam_gal[:2])   # eq.(8) of Cappellari (2017)

ppxf_dir = path.dirname(path.realpath(lib.__file__))
pathname = ppxf_dir + '/miles_models/Eun1.30*.fits'
FWHM_gal = None   # set this to None to skip convolutiona
miles = lib.miles(pathname, velscale, FWHM_gal, norm_range=[5070, 5950])
stars_templates, ln_lam_temp = miles.templates, miles.ln_lam_temp

reg_dim = stars_templates.shape[1:]
stars_templates = stars_templates.reshape(stars_templates.shape[0], -1)

stars_templates /= np.median(stars_templates) # Normalizes stellar templates by a scalar
vel0 = C*np.log(1 + redshift)  # Initial estimate of the galaxy velocity in km/s. eq. (8) of Cappellari (2017)
start = [vel0, 200.]  # (km/s), starting guess for [V,sigma]

lam_range_temp = np.exp(ln_lam_temp[[0, -1]])
lam_gal = np.exp(ln_lam_gal)
# fwhm_gal = 2.62  # Median FWHM resolution of MUSE
fwhm_gal = 2.75  # of the B&C spectrograph on the BOK Kitpeak obs see https://iopscience.iop.org/article/10.1086/500971/pdf
resolution = 8

gas_templates, gas_names, line_wave = util.emission_lines(miles.ln_lam_temp, lam_range_temp, fwhm_gal)
templates = np.column_stack([stars_templates, gas_templates])

n_stars = stars_templates.shape[1]
n_gas = len(gas_names)
component = [0]*n_stars + [1]*n_gas
gas_component = np.array(component) > 0  # gas_component=True for gas templates

moments = [7, 7]
start = [start, start]

pp = ppxf(templates, spectra_muse, spectra_muse_noise, velscale[0], start,
          moments=moments, degree=3, mdegree=5, lam=lam_gal, lam_temp=miles.lam_temp,
          reg_dim=reg_dim, component=component, gas_component=gas_component,
          reddening=0, gas_reddening=0, gas_names=gas_names)

# plt.figure(figsize=(17, 6))
# plt.subplot(111)
# pp.plot()
#
# # plt.show()
# plt.savefig('plot_output/em_fit_%s.png' % 'NGC_4214')
# plt.cla()

wavelength = pp.lam
total_flux = pp.galaxy
best_fit = pp.bestfit

gas_flux = pp.gas_flux
gas_flux_err = pp.gas_flux_error
gas_names = pp.gas_names

h_alpha_flux = gas_flux[gas_names == 'Halpha']
h_beta_flux = gas_flux[gas_names == 'Hbeta']
h_alpha_flux_err = gas_flux_err[gas_names == 'Halpha']
h_beta_flux_err = gas_flux_err[gas_names == 'Hbeta']

print('h_alpha_flux ', h_alpha_flux, h_alpha_flux_err)
print('h_beta_flux ', h_beta_flux, h_beta_flux_err)

balmer_dict = {
    'wavelength': wavelength,
    'total_flux': total_flux,
    'best_fit': best_fit,
    'h_alpha_flux': h_alpha_flux,
    'h_beta_flux': h_beta_flux,
    'h_alpha_flux_err': h_alpha_flux_err,
    'h_beta_flux_err': h_beta_flux_err,
}

np.save('data_output/balmer_dict_%s.npy' % 'NGC_4214', balmer_dict)

# plot spectra

# # get wsc projection

file_name_img = '/home/benutzer/data/observation/heII_paper/ngc4214/img/hst_11360_71_wfc3_uvis_f336w_drz.fits'
hdu_hst = fits.open(file_name_img)
print(hdu_hst.info())
header = hdu_hst['SCI'].header
img_data = hdu_hst['SCI'].data
wcs = WCS(header)

simbad_table = Simbad.query_object('NGC 4214')
central_coordinates = SkyCoord('%s %s' % (simbad_table['RA'].value[0], simbad_table['DEC'].value[0]),
                               unit=(u.hourangle, u.deg))

print('central_coordinates ', central_coordinates)

# load DSS image
paths_dss = SkyView.get_images(position='NGC 4214', survey='DSS2 IR', radius=10 *u.arcmin)
data_dss = paths_dss[0][0].data
wcs_dss = WCS(paths_dss[0][0].header)



# cutout_size = (200, 200)
# coords_img_world = SkyCoord('12h15m39.45s +36d19m34.6s', unit=(u.hourangle, u.deg))
# cutout_img = helper_func.get_img_cutout(img=img_data, wcs=wcs, coord=coords_img_world,
#                                               cutout_size=cutout_size)

# # create figure
fig = plt.figure(figsize=(20, 7))
# # snapshot
fontsize = 17
ax_image = fig.add_axes([-0.23, 0.1, 0.8, 0.8], projection=wcs_dss)
ax_h_beta = fig.add_axes([0.37, 0.09, 0.29, 0.89])
ax_h_alpha = fig.add_axes([0.70, 0.09, 0.29, 0.89])

print(np.nanmax(data_dss))
print(np.nanmedian(data_dss))

# ax_image.imshow(np.log10(data_dss), vmin=0.001, vmax=5, cmap='Greys')
ax_image.imshow(data_dss, vmin=-10, vmax=20000, cmap='Greys')
# spec_coords_world = SkyCoord('12h15m38.68s +36d20m27.89s', unit=(u.hourangle, u.deg))
spec_coords_pix = wcs_dss.world_to_pixel(central_coordinates)
slit_length = float(helper_func.transform_world2pix_scale(length_in_arcsec=190, wcs=wcs_dss))
slit_width = float(helper_func.transform_world2pix_scale(length_in_arcsec=120, wcs=wcs_dss))

ax_image.text(data_dss.shape[0]*0.03, data_dss.shape[1]*0.05, 'DSS',
              horizontalalignment='left', color='k', fontsize=fontsize)

plot_slit(ax=ax_image, coords_slit_pix=spec_coords_pix,
          slit_length=slit_length, slit_width=slit_width, slit_pa=float(90),
          plot_scatter=False, color='red', lw=3, linestyle='-')

plotting_tools.arr_axis_params(ax=ax_image, ra_tick_label=True, dec_tick_label=True,
                               ra_minpad=0.3, dec_minpad=0.8, tick_color='black',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=4, dec_tick_num=4)


mask_h_beta = (wavelength > 4830) & (wavelength < 4890)
mask_h_alpha = (wavelength > 6525) & (wavelength < 6619)

ax_h_beta.plot(wavelength[mask_h_beta], total_flux[mask_h_beta] * 1e15)
ax_h_beta.plot(wavelength[mask_h_beta], best_fit[mask_h_beta] * 1e15)
ax_h_alpha.plot(wavelength[mask_h_alpha], total_flux[mask_h_alpha] * 1e15)
ax_h_alpha.plot(wavelength[mask_h_alpha], best_fit[mask_h_alpha] * 1e15)


ax_h_beta.text(4863, 500, r'H$\beta$', fontsize=fontsize)
ax_h_alpha.text(6550, 1000, r'[NII]', fontsize=fontsize)
ax_h_alpha.text(6565, 1000, r'H$\alpha$', fontsize=fontsize)
ax_h_alpha.text(6585, 1000, r'[NII]', fontsize=fontsize)

ax_h_beta.set_ylabel('Flux [10$^{-15}$ erg/s/cm/cm/Å]', color='k', fontsize=fontsize)
ax_h_beta.set_xlabel('Observed Wavelength [Å]', color='k', fontsize=fontsize)
ax_h_alpha.set_xlabel('Observed Wavelength [Å]', color='k', fontsize=fontsize)

ax_h_beta.tick_params(axis='both', colors="k", labelsize=fontsize)
ax_h_alpha.tick_params(axis='both', colors="k", labelsize=fontsize)

plt.savefig('plot_output/em_fit_NGC_4214.png')


exit()


