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

C = 299792.458  # speed of light in km/s
# from scipy.constants import c as speed_of_light

cutout_size = (20, 20)


file_name_he2_10_muse = '/home/benutzer/data/observation/muse_data/he2_10/ADP.2016-06-17T18 13 44.227.fits'
hdu_muse = pyfits.open(file_name_he2_10_muse)
head_muse = hdu_muse[1].header
wcs_muse = WCS(head_muse)
cube_muse = hdu_muse[1].data   # cube.shape = (3681, nx, ny)
npix_muse = cube_muse.shape[0]
wave_muse = head_muse['CRVAL3'] + head_muse['CD3_3']*np.arange(npix_muse)
pixsize_muse = abs(head_muse["CD1_1"])*3600    # 0.2"

# npix = cube_muse.shape[0]
# spectra_muse = cube_muse.reshape(npix, -1) # create array of spectra_muse [npix, nx*ny]
# velscale_muse = C*np.diff(np.log(wave_muse[-2:]))  # Smallest velocity step
# lam_range_temp = [np.min(wave_muse), np.max(wave_muse)]
# spectra_muse, ln_lam_gal, velscale_muse = util.log_rebin(lam_range_temp, spectra_muse, velscale=velscale_muse)


file_name_he2_10_f555w = 'data/hst_06580_02_wfpc2_f555w_pc_drz.fits'
hdu_hst_f555w = pyfits.open(file_name_he2_10_f555w)
data_f555w = hdu_hst_f555w[1].data
wcs_f555w = WCS(hdu_hst_f555w[1].header)


coords_cent = SkyCoord('8h36m15.13s -26d24m33.7s', unit=(u.hourangle, u.deg))
coords_a = SkyCoord('8h36m15.2s -26d24m33.7s', unit=(u.hourangle, u.deg))
coords_b = SkyCoord('8h36m15.16s -26d24m33.7s', unit=(u.hourangle, u.deg))
coords_c = SkyCoord('8h36m15.14s -26d24m33.8s', unit=(u.hourangle, u.deg))
coords_d = SkyCoord('8h36m15.11s -26d24m33.9s', unit=(u.hourangle, u.deg))
coords_e = SkyCoord('8h36m14.81s -26d24m34.1s', unit=(u.hourangle, u.deg))


central_muse_pos_pix = wcs_muse.celestial.world_to_pixel(coords_cent)

print('central_muse_pos_pix ', central_muse_pos_pix)

print(cube_muse.shape)

x_lin_muse = np.linspace(1, cube_muse.shape[2], cube_muse.shape[2])
y_lin_muse = np.linspace(1, cube_muse.shape[1], cube_muse.shape[1])
x_data_muse, y_data_muse = np.meshgrid(x_lin_muse, y_lin_muse)

selection_radius_arcsec = 1
selection_radius_pix = helper_func.transform_world2pix_scale(length_in_arcsec=selection_radius_arcsec,
                                                             wcs=wcs_muse.celestial)

print('selection_radius_arcsec ', selection_radius_arcsec)
print('selection_radius_pix ', selection_radius_pix)
print(x_data_muse.shape)
print(y_data_muse.shape)

mask_spectrum = (np.sqrt((x_data_muse - central_muse_pos_pix[0]) ** 2 + (y_data_muse - central_muse_pos_pix[1]) ** 2)
                 < selection_radius_pix)

print('mask_spectrum ', mask_spectrum)
print('mask_spectrum ', mask_spectrum.shape)

print(cube_muse.shape)
lam_range_temp = [3540, 7409]   # Focus on optical regio
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

lam_gal = np.exp(ln_lam_gal)
print('velscale ', velscale)
# pp, bestfit_template = fit_and_clean(stars_templates, spectra_muse, velscale[0], start, goodpixels0, lam_gal, miles.lam_temp)

# pp = ppxf(stars_templates, spectra_muse, np.ones_like(spectra_muse), velscale[0], start,
#               moments=2, degree=-1, mdegree=6, lam=lam_gal, lam_temp=miles.lam_temp,
#               goodpixels=goodpixels0)
# pp.plot()
# plt.show()

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

moments = [2, 2]

start = [start, start]



pp = ppxf(templates, spectra_muse, np.ones_like(spectra_muse), velscale[0], start,
          moments=moments, degree=-1, mdegree=-1, lam=lam_gal, lam_temp=miles.lam_temp,
          reg_dim=reg_dim, component=component, gas_component=gas_component,
          reddening=0, gas_reddening=0, gas_names=gas_names)
plt.figure(figsize=(15, 5))
pp.plot()

plt.show()

exit()



lam_range_gal = [np.min(lam_gal), np.max(lam_gal)]
gas_templates, gas_names, line_wave = util.emission_lines(ln_lam_temp, lam_range_gal, fwhm_gal)

ngas_comp = 3    # I use three gas kinematic components
gas_templates = np.tile(gas_templates, ngas_comp)
gas_names = np.asarray([a + f"_({p+1})" for p in range(ngas_comp) for a in gas_names])
line_wave = np.tile(line_wave, ngas_comp)

stars_gas_templates = np.column_stack([stars_templates, gas_templates])


component = [0] + [1]*7 + [2]*7 + [3]*7
gas_component = np.array(component) > 0

moments = [-2, 2, 2, 2]

# start = [[V0, sig0], [V1, sig1], [V2, sig2], [V3, sig3]]
ncomp = len(moments)
tied = [['', ''] for j in range(ncomp)]
tied[2][1] = 'p[3]'                 # sig2 = sig1
tied[3][0] = '(p[2] + p[4])/2'      # V3 = (V1 + V2)/2



sig_diff = 200  # minimum dispersion difference in km/s
A_ineq = np.array([[0, 0, 0, 1, 0, 0, 0, -1],       # sigma2 - sigma4 < -sigma_diff
                   [0, 0, 0, 0, 0, 1, 0, -1]])      # sigma3 - sigma4 < -sigma_diff
b_ineq = np.array([-sig_diff, -sig_diff])/velscale  # velocities have to be expressed in pixels
constr_kinem = {"A_ineq": A_ineq, "b_ineq": b_ineq}

start = [[velbin[k], sigbin[k]],     # The stellar component=0 is fixed and this starting value will remain unchanged
         [velbin[k], 50],            # My starting guess for the velocity of all gas components is the velocity of the stars
         [velbin[k], 50],            # however, this starting guess is unnecessary and ignored when using `global_search=True`
         [velbin[k], 500]]           # The starting guess must be feasible, namely must satisfy the constraints


exit()

goodpixels = clip_outliers(galaxy, pp.bestfit, goodpixels0)

# Add clipped pixels to the original masked emission lines regions and repeat the fit
goodpixels = np.intersect1d(goodpixels, goodpixels0)
pp = ppxf(stars_templates, galaxy, np.ones_like(galaxy), velscale, start,
              moments=2, degree=-1, mdegree=4, lam=lam, lam_temp=lam_temp,
              goodpixels=goodpixels)

exit()


#
# plt.plot(lam, galaxy)
# plt.show()
#
# exit()

R = 2000
FWHM_gal = np.sqrt(1.66*3.17)/R
print( f"FWHM_gal: {FWHM_gal:.1f} Å")

c = 299792.458                      # speed of light in km/s
sigma_inst = c/(R*2.355)
print( f"sigma_inst: {sigma_inst:.0f} km/s")   # 47 km/s


z = 0.00283                       # Initial estimate of the galaxy redshift
lam /= (1 + z)               # Compute approximate restframe wavelength
FWHM_gal /= (1 + z)     # Adjust resolution in Angstrom
print(f"de-redshifted NIRSpec G235H/F170LP resolution FWHM in Å: {FWHM_gal:.1f}")

galaxy = galaxy/np.median(galaxy)       # Normalize spectrum to avoid numerical issues
noise = np.full_like(galaxy, 0.05)      # Assume constant noise per pixel here. I adopt a noise that gives chi2/DOF~1

velscale = c*np.log(lam[1]/lam[0])  # eq.(8) of Cappellari (2017)
print(f"Velocity scale per pixel: {velscale:.2f} km/s")

FWHM_temp = 2.51   # Resolution of E-MILES templates in the fitted range




ppxf_dir = path.dirname(path.realpath(lib.__file__))
pathname = ppxf_dir + '/miles_models/Eun1.30*.fits'
miles = lib.miles(pathname, velscale, norm_range=[5070, 5950], age_range=[0, 2.2])



reg_dim = miles.templates.shape[1:]
stars_templates = miles.templates.reshape(miles.templates.shape[0], -1)

lam_range_gal = [np.min(lam), np.max(lam)]
gas_templates, gas_names, line_wave = util.emission_lines(miles.ln_lam_temp, lam_range_gal, FWHM_gal, tie_balmer=1)


templates = np.column_stack([stars_templates, gas_templates])

c = 299792.458
start = [1200, 200.]     # (km/s), starting guess for [V, sigma]



n_stars = stars_templates.shape[1]
n_gas = len(gas_names)
component = [0]*n_stars + [1]*n_gas
gas_component = np.array(component) > 0  # gas_component=True for gas templates

moments = [2, 2]

start = [start, start]


pp = ppxf(templates, galaxy, noise, velscale, start,
          moments=moments, degree=-1, mdegree=-1, lam=lam, lam_temp=miles.lam_temp,
          reg_dim=reg_dim, component=component, gas_component=gas_component,
          reddening=0, gas_reddening=0, gas_names=gas_names)
plt.figure(figsize=(15, 5))
pp.plot()

plt.show()

exit()





print(spectrum.shape)
print(ln_lam_gal.shape)

# plt.plot(wave_muse, spectrum)
# plt.show()

z = 0.00283                       # Initial estimate of the galaxy redshift
lam = wave_muse(1 + z)
velscale = C*np.log(lam[1]/lam[0])  # eq.(8) of Cappellari (2017)

ppxf_dir = path.dirname(path.realpath(lib.__file__))
pathname = ppxf_dir + '/miles_models/Eun1.30*.fits'
miles = lib.miles(pathname, velscale, norm_range=[5070, 5950], age_range=[0, 2.2])

reg_dim = miles.templates.shape[1:]
stars_templates = miles.templates.reshape(miles.templates.shape[0], -1)

lam_range_gal = [np.min(lam), np.max(lam)]
gas_templates, gas_names, line_wave = util.emission_lines(miles.ln_lam_temp, lam_range_gal, FWHM_gal, tie_balmer=1)


ppxf_dir = path.dirname(path.realpath(lib.__file__))
pathname = ppxf_dir + '/miles_models/Eun1.30*.fits'
miles = lib.miles(pathname, velscale, norm_range=[5070, 5950])

pp = ppxf(templates, galaxy, noise, velscale, start,
          moments=moments, degree=-1, mdegree=-1, lam=wave_muse, lam_temp=miles.lam_temp,
          reg_dim=reg_dim, component=component, gas_component=gas_component,
          reddening=0, gas_reddening=0, gas_names=gas_names)





exit()



figure = plt.figure(figsize=(15, 15))

ax_muse = figure.add_axes([0.06, 0.04, 0.9, 0.93], projection=wcs_muse.celestial)
ax_muse.imshow(np.log10(np.nansum(cube_muse, 0)))
circle_muse = SphericalCircle(coords_cent, selection_radius_arcsec * u.arcsec, edgecolor='white', facecolor='None', linewidth=2, linestyle='-',
                               alpha=1.0, transform=ax_muse.get_transform('fk5'))
ax_muse.add_patch(circle_muse)
ax_muse.scatter(central_muse_pos_pix[0], central_muse_pos_pix[1])
plt.show()



exit()


cutout_hst = helper_func.get_img_cutout(img=data_f555w, wcs=wcs_f555w,
                                         coord=coords_cent, cutout_size=cutout_size)

cutout_muse = helper_func.get_img_cutout(img=np.nansum(cube_muse, 0), wcs=wcs_muse.celestial,
                                         coord=coords_cent, cutout_size=cutout_size)


figure = plt.figure(figsize=(20, 15))
fontsize = 23
ax_hst = figure.add_axes([0.06, 0.04, 0.45, 0.93], projection=cutout_hst.wcs)
ax_muse = figure.add_axes([0.52, 0.04, 0.45, 0.93], projection=cutout_muse.wcs)


ax_hst.imshow(np.log10(cutout_hst.data))
ax_muse.imshow(np.log10(cutout_muse.data))


circle_hst = SphericalCircle(coords_a, 0.2 * u.arcsec, edgecolor='white', facecolor='None', linewidth=2, linestyle='-',
                               alpha=1.0, transform=ax_hst.get_transform('fk5'))
ax_hst.add_patch(circle_hst)
circle_hst = SphericalCircle(coords_b, 0.2 * u.arcsec, edgecolor='white', facecolor='None', linewidth=2, linestyle='-',
                               alpha=1.0, transform=ax_hst.get_transform('fk5'))
ax_hst.add_patch(circle_hst)
circle_hst = SphericalCircle(coords_c, 0.2 * u.arcsec, edgecolor='white', facecolor='None', linewidth=2, linestyle='-',
                               alpha=1.0, transform=ax_hst.get_transform('fk5'))
ax_hst.add_patch(circle_hst)
circle_hst = SphericalCircle(coords_d, 0.2 * u.arcsec, edgecolor='white', facecolor='None', linewidth=2, linestyle='-',
                               alpha=1.0, transform=ax_hst.get_transform('fk5'))
ax_hst.add_patch(circle_hst)
circle_hst = SphericalCircle(coords_e, 0.2 * u.arcsec, edgecolor='white', facecolor='None', linewidth=2, linestyle='-',
                               alpha=1.0, transform=ax_hst.get_transform('fk5'))
ax_hst.add_patch(circle_hst)


circle_muse = SphericalCircle(coords_a, 0.2 * u.arcsec, edgecolor='white', facecolor='None', linewidth=2, linestyle='-',
                               alpha=1.0, transform=ax_muse.get_transform('fk5'))
ax_muse.add_patch(circle_muse)


plt.show()


exit()



class read_muse_cube(object):
    def __init__(self, wave_range):
        """
        Read MUSE cube, log rebin it and compute coordinates of each spaxel.
        Median FWHM resolution = 2.62Å. Range: 2.51--2.88 (ESOpPXF Purpose
        instrument manual)

        """
        # filename = 'LVS_JWST_workshop_rodeo_cube.fits'  # NGC1386
        filename = '/home/benutzer/data/observation/muse_data/he2_10/ADP.2016-06-17T18 13 44.227.fits'
        hdu = pyfits.open(filename)

        head = hdu[1].header
        cube = hdu[1].data   # cube.shape = (3681, nx, ny)
        #print('header ', head)
        print(cube.shape)

        # Transform cube into 2-dim array of spectra
        npix = cube.shape[0]
        spectra = cube.reshape(npix, -1) # create array of spectra [npix, nx*ny]

        print(spectra.shape)

        plt.imshow(np.log10(np.nansum(cube, 0)))
        plt.show()


        exit()

        # wave = head['CRVAL3'] + head['CDELT3']*np.arange(npix)
        wave = head['CRVAL3'] + head['CD3_3']*np.arange(npix)
        # pixsize = abs(head["CDELT1"])*3600    # 0.2"
        pixsize = abs(head["CD1_1"])*3600    # 0.2"

        # Only use a restricted wavelength range
        w = (wave > wave_range[0]) & (wave < wave_range[1])
        spectra = spectra[w, :]
        wave = wave[w]

        # Create coordinates centred on the brightest spectrum
        flux = np.nanmean(spectra, 0)
        jm = np.nanargmax(flux)

        # print('jm ', jm)
        # plt.plot(wave, spectra[:, jm])
        # plt.show()
        #
        # exit()

        row, col = map(np.ravel, np.indices(cube.shape[-2:]))
        x = (col - col[jm])*pixsize
        y = (row - row[jm])*pixsize
        velscale = C*np.diff(np.log(wave[-2:]))  # Smallest velocity step
        lam_range_temp = [np.min(wave), np.max(wave)]
        spectra, ln_lam_gal, velscale = util.log_rebin(lam_range_temp, spectra, velscale=velscale)

        self.spectra = spectra
        self.x = x
        self.y = y
        self.col = col + 1   # start counting from 1
        self.row = row + 1
        self.flux = flux
        self.ln_lam_gal = ln_lam_gal
        self.fwhm_gal = 2.62  # Median FWHM resolution of MUSE

        print('spectra ', self.spectra)
        print('x ', self.x)
        print('y ', self.y)
        print('col ', self.col)
        print('row ', self.row)
        print('flux ', self.flux)
        print('ln_lam_gal ', self.ln_lam_gal)
        print('fwhm_gal ', self.fwhm_gal)


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
              moments=2, degree=-1, mdegree=4, lam=lam, lam_temp=lam_temp,
              goodpixels=goodpixels)

    plt.figure(figsize=(20, 3))
    plt.subplot(121)
    pp.plot()

    goodpixels = clip_outliers(galaxy, pp.bestfit, goodpixels)

    # Add clipped pixels to the original masked emission lines regions and repeat the fit
    goodpixels = np.intersect1d(goodpixels, goodpixels0)
    pp = ppxf(templates, galaxy, np.ones_like(galaxy), velscale, start,
              moments=2, degree=-1, mdegree=4, lam=lam, lam_temp=lam_temp,
              goodpixels=goodpixels)

    plt.subplot(122)
    pp.plot()

    optimal_template = templates @ pp.weights

    return pp, optimal_template

lam_range_temp = [3540, 7409]   # Focus on optical region
s = read_muse_cube(lam_range_temp)


signal = np.nanmedian(s.spectra, 0)
signal[np.isnan(signal)] = 1e-6
# print(sum(np.isnan(signal)))
#
# print(signal.shape)
#
# exit()


noise = np.sqrt(abs(signal))
target_sn = 350
# target_sn = 60


plt.figure(figsize=(7,10))
bin_num, x_gen, y_gen, xbin, ybin, sn, nPixels, scale = voronoi_2d_binning(s.x, s.y, signal, noise, target_sn, plot=1, quiet=1)
plt.show()

