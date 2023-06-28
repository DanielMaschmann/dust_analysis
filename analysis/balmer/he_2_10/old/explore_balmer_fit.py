import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.io import fits
from photometry_tools.analysis_tools import AnalysisTools
from photometry_tools import helper_func, plotting_tools
import astropy.io.fits as pyfits
import numpy as np
import matplotlib.pyplot as plt
from os import path

from ppxf.ppxf import ppxf, robust_sigma
import ppxf.ppxf_util as util
import ppxf.miles_util as lib
from vorbin.voronoi_2d_binning import voronoi_2d_binning
from plotbin.display_bins import display_bins
from plotbin.plot_velfield import plot_velfield

from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from photometry_tools.analysis_tools import AnalysisTools
from photometry_tools import helper_func, plotting_tools
from astropy.visualization.wcsaxes import SphericalCircle
from matplotlib.patches import ConnectionPatch

import dust_tools

C = 299792.458  # speed of light in km/s
# from scipy.constants import c as speed_of_light



def clip_outliers(galaxy, bestfit, goodpixels):
    """
    Repeat the fit after clipping bins deviants more than 3*sigma
    in relative error until the bad bins don't change any more.
    """
    while True:
        scale = galaxy[goodpixels] @ bestfit[goodpixels]/np.sum(bestfit[goodpixels]**2)
        resid = scale*bestfit[goodpixels] - galaxy[goodpixels]
        err = robust_sigma(resid, zero=1)
        ok_old = goodpixels
        goodpixels = np.flatnonzero(np.abs(bestfit - galaxy) < 3*err)
        if np.array_equal(goodpixels, ok_old):
            break

    return goodpixels

def fit_and_clean(templates, galaxy, velscale, start, goodpixels0, lam, lam_temp):

    print('##############################################################')
    goodpixels = goodpixels0.copy()
    pp = ppxf(templates, galaxy, np.ones_like(galaxy), velscale, start,
              moments=2, degree=3, mdegree=6, lam=lam, lam_temp=lam_temp,
              goodpixels=goodpixels)

    plt.figure(figsize=(20, 3))
    plt.subplot(121)
    pp.plot()
    goodpixels = clip_outliers(galaxy, pp.bestfit, goodpixels)

    # Add clipped pixels to the original masked emission lines regions and repeat the fit
    goodpixels = np.intersect1d(goodpixels, goodpixels0)
    pp = ppxf(templates, galaxy, np.ones_like(galaxy), velscale, start,
              moments=2, degree=3, mdegree=6, lam=lam, lam_temp=lam_temp,
              goodpixels=goodpixels)

    plt.subplot(122)
    pp.plot()
    plt.show()

    optimal_template = templates @ pp.weights

    return pp, optimal_template


file_name_he2_10_muse = '/home/benutzer/data/observation/muse_data/he2_10/ADP.2016-06-17T18 13 44.227.fits'
hdu_muse = fits.open(file_name_he2_10_muse)
head_muse = hdu_muse[1].header
wcs_muse = WCS(head_muse)
cube_muse = hdu_muse[1].data   # cube.shape = (3681, nx, ny)
npix_muse = cube_muse.shape[0]
wave_muse = head_muse['CRVAL3'] + head_muse['CD3_3']*np.arange(npix_muse)
pixsize_muse = abs(head_muse["CD1_1"])*3600    # 0.2"
mask_h_alpha = (wave_muse > 6570) & (wave_muse < 6590)
data_muse_h_alpha = np.nansum(cube_muse[mask_h_alpha, :, :], 0)


coords_a = SkyCoord('8h36m15.199s -26d24m33.68s', unit=(u.hourangle, u.deg))
coords_b = SkyCoord('8h36m15.166s -26d24m33.75s', unit=(u.hourangle, u.deg))
coords_c = SkyCoord('8h36m15.136s -26d24m33.78s', unit=(u.hourangle, u.deg))
coords_d = SkyCoord('8h36m15.114s -26d24m33.84s', unit=(u.hourangle, u.deg))



x_lin_muse = np.linspace(1, cube_muse.shape[2], cube_muse.shape[2])
y_lin_muse = np.linspace(1, cube_muse.shape[1], cube_muse.shape[1])
x_data_muse, y_data_muse = np.meshgrid(x_lin_muse, y_lin_muse)

selection_radius_arcsec = 0.2
selection_radius_pix = helper_func.transform_world2pix_scale(length_in_arcsec=selection_radius_arcsec,
                                                             wcs=wcs_muse.celestial)

print('selection_radius_arcsec ', selection_radius_arcsec)
print('selection_radius_pix ', selection_radius_pix)
print(x_data_muse.shape)
print(y_data_muse.shape)

coords_a_muse_pos_pix = wcs_muse.celestial.world_to_pixel(coords_d)

mask_spectrum = (np.sqrt((x_data_muse - coords_a_muse_pos_pix[0]) ** 2 + (y_data_muse - coords_a_muse_pos_pix[1]) ** 2)
                 < selection_radius_pix)

print('mask_spectrum ', mask_spectrum)
print('mask_spectrum ', mask_spectrum.shape)

print(cube_muse.shape)
lam_range_temp = [3540, 8009]   # Focus on optical regio
galaxy = np.sum(cube_muse[:, mask_spectrum], 1)
lam = wave_muse
w = (lam > lam_range_temp[0]) & (lam < lam_range_temp[1])
galaxy = galaxy[w]
lam = lam[w]


lam_range_temp = [np.min(lam), np.max(lam)]
velscale = C*np.diff(np.log(lam[-2:]))  # Smallest velocity step
spectra_muse, ln_lam_gal, velscale_muse = util.log_rebin(lam_range_temp, galaxy, velscale=velscale)

print('spectra_muse ', spectra_muse)
print('ln_lam_gal ', ln_lam_gal)
print('velscale_muse ', velscale_muse)

velscale = C*np.diff(ln_lam_gal[:2])   # eq.(8) of Cappellari (2017)


ppxf_dir = path.dirname(path.realpath(lib.__file__))
pathname = ppxf_dir + '/miles_models/Eun1.30*.fits'
FWHM_gal = None   # set this to None to skip convolutiona
miles = lib.miles(pathname, velscale, FWHM_gal, norm_range=[5070, 5950])
stars_templates, ln_lam_temp = miles.templates, miles.ln_lam_temp


reg_dim = stars_templates.shape[1:]
stars_templates = stars_templates.reshape(stars_templates.shape[0], -1)

stars_templates /= np.median(stars_templates) # Normalizes stellar templates by a scalar
regul_err = 0.01 # Desired regularization error

z = 0.00283   # redshift estimate from NED
vel0 = C*np.log(1 + z)  # Initial estimate of the galaxy velocity in km/s. eq. (8) of Cappellari (2017)
start = [vel0, 200.]  # (km/s), starting guess for [V,sigma]

lam_range_temp = np.exp(ln_lam_temp[[0, -1]])
goodpixels0 = util.determine_goodpixels(ln_lam_gal, lam_range_temp, z, width=1000)


print('goodpixels0 ', goodpixels0)


lam_gal = np.exp(ln_lam_gal)
print('velscale ', velscale)

# pp, bestfit_template = fit_and_clean(stars_templates, spectra_muse, velscale[0], start, goodpixels0, lam_gal, miles.lam_temp)

plt.figure(figsize=(17, 6))
plt.subplot(111)
pp = ppxf(stars_templates, spectra_muse, np.ones_like(spectra_muse), velscale[0], start,
              moments=2, degree=7, mdegree=10, lam=lam_gal, lam_temp=miles.lam_temp,
              goodpixels=goodpixels0)
pp.plot()
# plt.show()
plt.savefig('plot_output/stellar_fit_1.png')
plt.cla()

fwhm_gal = 2.62  # Median FWHM resolution of MUSE


print('miles.ln_lam_temp ', miles.ln_lam_temp)
print('lam_range_temp ', lam_range_temp)
print('FWHM_gal ', FWHM_gal)

gas_templates, gas_names, line_wave = util.emission_lines(miles.ln_lam_temp, lam_range_temp, fwhm_gal)

templates = np.column_stack([stars_templates, gas_templates])

n_stars = stars_templates.shape[1]
n_gas = len(gas_names)
component = [0]*n_stars + [1]*n_gas
gas_component = np.array(component) > 0  # gas_component=True for gas templates

moments = [3, 3]

start = [start, start]

pp = ppxf(templates, spectra_muse, np.ones_like(spectra_muse), velscale[0], start,
          moments=moments, degree=-1, mdegree=-1, lam=lam_gal, lam_temp=miles.lam_temp,
          reg_dim=reg_dim, component=component, gas_component=gas_component,
          reddening=0, gas_reddening=0, gas_names=gas_names)

plt.figure(figsize=(17, 6))
plt.subplot(111)
pp.plot()

# plt.show()
plt.savefig('plot_output/em_fit_1.png')
plt.cla()


gas_flux = pp.gas_flux
gas_flux_err = pp.gas_flux_error
gas_names = pp.gas_names

h_alpha_flux = gas_flux[gas_names == 'Halpha']
h_beta_flux = gas_flux[gas_names == 'Hbeta']

print('h_alpha_flux ', h_alpha_flux)
print('h_beta_flux ', h_beta_flux)
# claculate E(B-V)
dust_class = dust_tools.extinction_tools.ExtinctionTools()

ebv_balmer = dust_class.get_balmer_extinct_alpha_beta(flux_h_alpha_6565=h_alpha_flux,
                                           flux_h_beta_4863=h_beta_flux)

print('ebv_balmer ', ebv_balmer)

