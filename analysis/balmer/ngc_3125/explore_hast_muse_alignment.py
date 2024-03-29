import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.visualization.wcsaxes import SphericalCircle
from astropy.io import fits
from photometry_tools.analysis_tools import AnalysisTools
from photometry_tools import helper_func, plotting_tools
import multicolorfits as mcf
from astropy.wcs.utils import fit_wcs_from_points


file_name_ngc3125_muse = '/home/benutzer/data/observation/heII_paper/ngc3125/muse/ADP.2016-06-03T11 20 45.454.fits'
hdu_muse = fits.open(file_name_ngc3125_muse)
head_muse = hdu_muse[1].header
wcs_muse = WCS(head_muse)
cube_muse = hdu_muse[1].data   # cube.shape = (3681, nx, ny)
cube_muse_err = hdu_muse[2].data   # cube.shape = (3681, nx, ny)
npix_muse = cube_muse.shape[0]
wave_muse = head_muse['CRVAL3'] + head_muse['CD3_3']*np.arange(npix_muse)
pixsize_muse = abs(head_muse["CD1_1"])*3600    # 0.2"
mask_h_alpha = (wave_muse > 6570) & (wave_muse < 6590)
data_muse_h_alpha = np.sum(cube_muse[mask_h_alpha, :, :], 0)
data_muse_h_alpha_err = np.sum(cube_muse_err[mask_h_alpha, :, :], 0)



file_name_ngc3125_f555w = '/home/benutzer/data/observation/heII_paper/ngc3125/img/hst_10400_01_acs_hrc_f555w_drz.fits'
file_name_ngc3125_f814w = '/home/benutzer/data/observation/heII_paper/ngc3125/img/hst_10400_01_acs_hrc_f814w_drz.fits'
file_name_ngc3125_f658n = '/home/benutzer/data/observation/heII_paper/ngc3125/img/hst_10400_01_acs_hrc_f658n_drz.fits'
hdu_hst_f555w = fits.open(file_name_ngc3125_f555w)
hdu_hst_f814w = fits.open(file_name_ngc3125_f814w)
hdu_hst_f658n = fits.open(file_name_ngc3125_f658n)
data_f555w = hdu_hst_f555w[1].data
data_f814w = hdu_hst_f814w[1].data
data_f658n = hdu_hst_f658n[1].data
wcs_f555w = WCS(hdu_hst_f555w[1].header)
wcs_f814w = WCS(hdu_hst_f814w[1].header)
wcs_f658n = WCS(hdu_hst_f658n[1].header)


# figure = plt.figure(figsize=(30, 15))
# fontsize = 23
# ax_hst = figure.add_axes([0.06, 0.04, 0.45, 0.93], projection=wcs_f658n)
# ax_muse = figure.add_axes([0.52, 0.04, 0.45, 0.93]) #, projection=wcs_muse.celestial)
#
#
# ax_hst.imshow(np.log10(data_f658n), cmap='Greys')
# ax_muse.imshow(np.log10(data_muse_h_alpha), origin='lower', cmap='Greys')
#
# plt.show()
#
# exit()


# # update MUSE
stars = [np.array([188.60, 127.98, 86.22]), np.array([175.52, 118.63, 112.84])]
known_coords = SkyCoord(['10h06m32.512s -29d56m03.60s', '10h06m33.441s -29d56m14.83s','10h06m34.10s -29d56m16.2s'], unit=(u.hourangle, u.deg))
new_muse_wcs = fit_wcs_from_points(xy=stars, world_coords=known_coords, projection=wcs_muse.celestial)
# new_muse_wcs = wcs_muse.celestial



cutout_size = (12, 12)
coords_cent = SkyCoord('10h06m33.261s -29d56m08.52s', unit=(u.hourangle, u.deg))
cutout_f555w = helper_func.get_img_cutout(img=data_f555w, wcs=wcs_f555w, coord=coords_cent, cutout_size=cutout_size)
cutout_f814w = helper_func.get_img_cutout(img=data_f814w, wcs=wcs_f814w, coord=coords_cent, cutout_size=cutout_size)
cutout_f658n = helper_func.get_img_cutout(img=data_f658n, wcs=wcs_f658n, coord=coords_cent, cutout_size=cutout_size)
cutout_muse = helper_func.get_img_cutout(img=data_muse_h_alpha, wcs=new_muse_wcs, coord=coords_cent, cutout_size=cutout_size)


# # reproject hst_cutouts to f555w
new_wcs = cutout_f555w.wcs
new_shape = cutout_f555w.data.shape
hst_img_g = plotting_tools.reproject_image(data=cutout_f814w.data,
                                           wcs=cutout_f814w.wcs,
                                           new_wcs=new_wcs,
                                           new_shape=new_shape)
hst_img_r = plotting_tools.reproject_image(data=cutout_f658n.data,
                                           wcs=cutout_f658n.wcs,
                                           new_wcs=new_wcs,
                                           new_shape=new_shape)

grey_hst_r = mcf.greyRGBize_image(hst_img_r, rescalefn='asinh', scaletype='perc', min_max=[1.0, 99.9], gamma=1.5, checkscale=False)
grey_hst_g = mcf.greyRGBize_image(hst_img_g, rescalefn='asinh', scaletype='perc', min_max=[1.0, 99.9], gamma=2.8, checkscale=False)
grey_hst_b = mcf.greyRGBize_image(cutout_f555w.data, rescalefn='asinh', scaletype='perc', min_max=[1.0, 99.9], gamma=2.8, checkscale=False)
hst_r_purple = mcf.colorize_image(grey_hst_r, '#FF4433', colorintype='hex', gammacorr_color=2.2)
hst_g_orange = mcf.colorize_image(grey_hst_g, '#4CBB17', colorintype='hex', gammacorr_color=2.2)
hst_b_blue = mcf.colorize_image(grey_hst_b, '#0096FF', colorintype='hex', gammacorr_color=2.2)
rgb_hst_image = mcf.combine_multicolor([hst_r_purple, hst_g_orange, hst_b_blue], gamma=1.5, inverse=False)

#
coords_a_hst = SkyCoord('10h06m33.263s -29d56m06.76s', unit=(u.hourangle, u.deg))
# coords_b_hst = SkyCoord('8h36m15.166s -26d24m33.75s', unit=(u.hourangle, u.deg))
# coords_c_hst = SkyCoord('8h36m15.136s -26d24m33.78s', unit=(u.hourangle, u.deg))
# coords_d_hst = SkyCoord('8h36m15.114s -26d24m33.84s', unit=(u.hourangle, u.deg))
#
color_a = 'tab:blue'
# color_b = 'tab:orange'
# color_c = 'tab:green'
# color_d = 'tab:red'

figure = plt.figure(figsize=(30, 15))
fontsize = 23
ax_hst = figure.add_axes([0.06, 0.04, 0.45, 0.93], projection=new_wcs)
ax_muse = figure.add_axes([0.52, 0.04, 0.45, 0.93], projection=cutout_muse.wcs)
ax_balmer_1 = figure.add_axes([0.31, 0.12, 0.29, 0.3])
ax_balmer_2 = figure.add_axes([0.61, 0.12, 0.29, 0.3])


ax_hst.imshow(rgb_hst_image)
ax_muse.imshow(np.log10(cutout_muse.data), cmap='Greys')

circle_hst = SphericalCircle(coords_a_hst, 0.2 * u.arcsec, edgecolor=color_a, facecolor='None', linewidth=2, linestyle='-',
                               alpha=1.0, transform=ax_hst.get_transform('fk5'))
ax_hst.add_patch(circle_hst)

circle_muse = SphericalCircle(coords_a_hst, 0.5 * u.arcsec, edgecolor=color_a, facecolor='None', linewidth=2, linestyle='-',
                               alpha=1.0, transform=ax_muse.get_transform('fk5'))
ax_muse.add_patch(circle_muse)


pix_a = new_wcs.world_to_pixel(coords_a_hst)
ax_hst.text(pix_a[0] - 3, pix_a[1] + 3, 'A', color=color_a, horizontalalignment='right', verticalalignment='bottom', fontsize=fontsize)

pix_a = cutout_muse.wcs.world_to_pixel(coords_a_hst)
ax_muse.text(pix_a[0] - 1.5, pix_a[1] + 1.5, 'A', color=color_a, horizontalalignment='right', verticalalignment='bottom', fontsize=fontsize)


ax_hst.text(new_shape[0]*0.03, new_shape[1]*0.96, 'HST', horizontalalignment='left', color='white', fontsize=fontsize)
ax_hst.text(new_shape[0]*0.03, new_shape[1]*0.93, 'F555W', horizontalalignment='left', color='b', fontsize=fontsize)
ax_hst.text(new_shape[0]*0.03, new_shape[1]*0.90, 'F814W', horizontalalignment='left', color='g', fontsize=fontsize)
ax_hst.text(new_shape[0]*0.03, new_shape[1]*0.87, r'F658N H$\alpha$', horizontalalignment='left', color='r', fontsize=fontsize)

ax_muse.text(cutout_muse.data.shape[0]*0.03, cutout_muse.data.shape[1]*0.95, r'MUSE H$\alpha$', horizontalalignment='left', color='k', fontsize=fontsize)

ax_hst.set_title('NGC 3125', fontsize=fontsize)
plotting_tools.arr_axis_params(ax=ax_hst, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label='R.A. (2000.0)', dec_axis_label='DEC. (2000.0)',
                               ra_minpad=0.3, dec_minpad=0.8, tick_color='black',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=4, dec_tick_num=4)

plotting_tools.arr_axis_params(ax=ax_muse, ra_tick_label=True, dec_tick_label=False,
                               ra_axis_label='R.A. (2000.0)', dec_axis_label='',
                               ra_minpad=0.3, dec_minpad=0.8, tick_color='black',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=4, dec_tick_num=4)


balmer_dict_a = np.load('data_output/balmer_dict_NGC3125A.npy', allow_pickle=True).item()

ax_balmer_1.plot(balmer_dict_a['wavelength'], balmer_dict_a['total_flux'] * 1e-5, color=color_a)
ax_balmer_2.plot(balmer_dict_a['wavelength'], balmer_dict_a['total_flux'] * 1e-5, color=color_a)

ax_balmer_1.plot([4863 + 16, 4863 + 16], [-2, 28], linestyle='--', color='k')
ax_balmer_1.plot([4960 + 17, 4960 + 17], [-2, 28], linestyle='--', color='k')
ax_balmer_1.plot([5008 + 17, 5008 + 17], [-2, 28], linestyle='--', color='k')

ax_balmer_2.plot([6550 + 21.5, 6550 + 21.5], [-2, 28], linestyle='--', color='k')
ax_balmer_2.plot([6565 + 21.5, 6565 + 21.5], [-2, 28], linestyle='--', color='k')
ax_balmer_2.plot([6585 + 21.5, 6585 + 21.5], [-2, 28], linestyle='--', color='k')


ax_balmer_1.text(4863 + 16, 12.0, r'H$\beta$', fontsize=fontsize)
ax_balmer_1.text(4960 + 17, 12.0, r'[OIII]', fontsize=fontsize)
ax_balmer_1.text(5008 + 17, 12.0, r'[OIII]', fontsize=fontsize)

ax_balmer_2.text(6550 + 21.5, 12.0, r'[NII]', fontsize=fontsize)
ax_balmer_2.text(6565 + 21.5, 12.0, r'H$\alpha$', fontsize=fontsize)
ax_balmer_2.text(6585 + 21.5, 12.0, r'[NII]', fontsize=fontsize)

ax_balmer_1.set_xlim(4846, 5054)
ax_balmer_2.set_xlim(6561, 6616)

ax_balmer_1.set_ylim(-0.9, 13)
ax_balmer_2.set_ylim(-0.9, 13)


# hide the spines between ax and ax2
ax_balmer_1.spines['right'].set_visible(False)
ax_balmer_2.spines['left'].set_visible(False)
# ax_balmer_1.yaxis.tick_left()
# ax_balmer_1.tick_params(labelright='off')
ax_balmer_2.yaxis.tick_right()

ax_balmer_1.set_xticks([4850, 4875, 4900, 4925, 4950, 4975])
ax_balmer_1.set_ylabel('Flux [10$^{-15}$ erg/s/cm/cm/Å]', color='white', fontsize=fontsize)
ax_balmer_1.set_xlabel('Observed Wavelength [Å]                       ', color='white', fontsize=fontsize)

ax_balmer_2.set_xlabel('Observed Wavelength [Å]', fontsize=fontsize)

# ax_balmer_1.spines['left'].set_color('white')
ax_balmer_1.tick_params(axis='both', colors="white", labelsize=fontsize)
ax_balmer_2.tick_params(axis='both', colors="black", labelsize=fontsize)

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=ax_balmer_1.transAxes, color='k', clip_on=False)
ax_balmer_1.plot((1-d, 1+d), (-d, +d), **kwargs)
ax_balmer_1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

kwargs.update(transform=ax_balmer_2.transAxes)  # switch to the bottom axes
ax_balmer_2.plot((-d, +d), (1-d, 1+d), **kwargs)
ax_balmer_2.plot((-d, +d), (-d, +d), **kwargs)


# plt.show()
plt.savefig('plot_output/hst_muse_overview.png')


