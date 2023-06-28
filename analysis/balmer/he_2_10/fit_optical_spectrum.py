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



def fit_muse_spectrum(coords, select_rad_pix, redshift, target_name):

    coords_muse_pos_pix = new_muse_wcs.world_to_pixel(coords)

    mask_spectrum = (np.sqrt((x_data_muse - coords_muse_pos_pix[0]) ** 2 +
                             (y_data_muse - coords_muse_pos_pix[1]) ** 2) < select_rad_pix)
    print('mask_spectrum ', np.sum(mask_spectrum))
    print('mask_spectrum ', mask_spectrum.shape)

    lam_range_temp = [3540, 8009]   # Focus on optical regio
    galaxy = np.sum(cube_muse[:, mask_spectrum], 1)
    lam = wave_muse
    w = (lam > lam_range_temp[0]) & (lam < lam_range_temp[1])
    galaxy = galaxy[w]
    lam = lam[w]

    lam_range_temp = [np.min(lam), np.max(lam)]
    velscale = C*np.diff(np.log(lam[-2:]))  # Smallest velocity step
    spectra_muse, ln_lam_gal, velscale_muse = util.log_rebin(lam_range_temp, galaxy, velscale=velscale)

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
    fwhm_gal = 2.62  # Median FWHM resolution of MUSE

    # print('miles.ln_lam_temp ', miles.ln_lam_temp)
    # print('lam_range_temp ', lam_range_temp)
    # print('FWHM_gal ', FWHM_gal)

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
    plt.savefig('plot_output/em_fit_%s.png' % target_name)
    plt.cla()

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

    np.save('data_output/balmer_dict_%s.npy' % target_name, balmer_dict)


file_name_he2_10_muse = '/home/benutzer/data/observation/heII_paper/he2_10/muse/ADP.2016-06-17T18 13 44.227.fits'
hdu_muse = fits.open(file_name_he2_10_muse)
head_muse = hdu_muse[1].header
wcs_muse = WCS(head_muse)
cube_muse = hdu_muse[1].data   # cube.shape = (3681, nx, ny)
npix_muse = cube_muse.shape[0]
wave_muse = head_muse['CRVAL3'] + head_muse['CD3_3']*np.arange(npix_muse)
pixsize_muse = abs(head_muse["CD1_1"])*3600    # 0.2"

# update MUSE
stars = [np.array([36.68, 179.37, 150.705]), np.array([ 177.23, 131.73, 218.259])]
known_coords = SkyCoord(['8h36m17.013s -26d24m32.18s', '8h36m14.886s -26d24m41.31s', '8h36m15.302s -26d24m24.0s'], unit=(u.hourangle, u.deg))
new_muse_wcs = fit_wcs_from_points(xy=stars, world_coords=known_coords, projection=wcs_muse.celestial)


x_lin_muse = np.linspace(1, cube_muse.shape[2], cube_muse.shape[2])
y_lin_muse = np.linspace(1, cube_muse.shape[1], cube_muse.shape[1])
x_data_muse, y_data_muse = np.meshgrid(x_lin_muse, y_lin_muse)


selection_radius_arcsec = 0.2
selection_radius_pix = helper_func.transform_world2pix_scale(length_in_arcsec=selection_radius_arcsec,
                                                             wcs=new_muse_wcs)
print('selection_radius_arcsec ', selection_radius_arcsec)
print('selection_radius_pix ', selection_radius_pix)

coords_a = SkyCoord('8h36m15.199s -26d24m33.68s', unit=(u.hourangle, u.deg))
coords_b = SkyCoord('8h36m15.166s -26d24m33.75s', unit=(u.hourangle, u.deg))
coords_c = SkyCoord('8h36m15.136s -26d24m33.78s', unit=(u.hourangle, u.deg))
coords_d = SkyCoord('8h36m15.114s -26d24m33.84s', unit=(u.hourangle, u.deg))
redshift = 0.00283   # redshift estimate from NED


fit_muse_spectrum(coords=coords_a, select_rad_pix=selection_radius_pix, redshift=redshift, target_name='HE_2_10A')
fit_muse_spectrum(coords=coords_b, select_rad_pix=selection_radius_pix, redshift=redshift, target_name='HE_2_10B')
fit_muse_spectrum(coords=coords_c, select_rad_pix=selection_radius_pix, redshift=redshift, target_name='HE_2_10C')
fit_muse_spectrum(coords=coords_d, select_rad_pix=selection_radius_pix, redshift=redshift, target_name='HE_2_10D')
