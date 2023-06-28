import numpy as np
from astropy.io import fits
import astropy.wcs as wcs
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from photometry_tools import helper_func, plotting_tools, analysis_tools
from regions import PixCoord, RectanglePixelRegion
from astropy.stats import sigma_clipped_stats


def plot_slit(ax, coords_slit_pix, slit_length, slit_width, slit_pa, plot_scatter=True, color='red', lw=4, linestyle='-'):
    x_cent = float(coords_slit_pix[0])
    y_cent = float(coords_slit_pix[1])

    reg = RectanglePixelRegion(PixCoord(x=x_cent, y=y_cent), width=slit_length,
                               height=slit_width, angle=slit_pa*u.deg)
    reg.plot(ax=ax, edgecolor=color, linewidth=lw, linestyle=linestyle)
    if plot_scatter:
        ax.scatter(coords_slit_pix[0], coords_slit_pix[1], c='r', s=120)


target = 'tol1924_416'

file_name_dict = {
    'tol1924_416': {
        'file_name_img': '/home/benutzer/data/observation/heII_paper/tol1924_416/img/hst_06708_01_wfpc2_f439w_pc_drz.fits',
        'filter_name_img': 'F439W',
        'file_name_align_g140l': '/home/benutzer/data/observation/heII_paper/tol1924_416/stis/O6ER18PEQ/o6er18peq_raw.fits',
        'file_name_stis_g140l': '/home/benutzer/data/observation/heII_paper/tol1924_416/stis/O6ER18020/o6er18020_flt.fits',
        'file_name_align_g430m': '/home/benutzer/data/observation/heII_paper/tol1924_416/stis/OE0708R7Q/oe0708r7q_raw.fits',
        'file_name_stis_g430m': '/home/benutzer/data/observation/heII_paper/tol1924_416/stis/OE0708020/oe0708020_flt.fits',
        'coords_img': '19h27m58.0s -41d34m31.0s',
        'coords_slit_center': '19h27m58.382s -41d34m30.94s',
        'coords_source_a': '19h27m58.382s -41d34m30.94s',
        #'coords_source_b': '19h27m57.90s -41d34m31.37s',
    }}


# load slit_observations
hdu_stis_g140l = fits.open(file_name_dict[target]['file_name_stis_g140l'])
hdu_align_g140l = fits.open(file_name_dict[target]['file_name_align_g140l'])

hdu_stis_g430m = fits.open(file_name_dict[target]['file_name_stis_g430m'])
hdu_align_g430m = fits.open(file_name_dict[target]['file_name_align_g430m'])


wcs_g140l_1 = wcs.WCS(hdu_align_g140l[1].header)
wcs_g140l_2 = wcs.WCS(hdu_align_g140l[4].header)
wcs_g140l_3 = wcs.WCS(hdu_align_g140l[7].header)

wcs_g430m_1 = wcs.WCS(hdu_align_g430m[1].header)
wcs_g430m_2 = wcs.WCS(hdu_align_g430m[4].header)
wcs_g430m_3 = wcs.WCS(hdu_align_g430m[7].header)


aperture_stis_g140l = hdu_stis_g140l[0].header['APERTURE']
slit_length_g140l = float(aperture_stis_g140l.split('X')[0])
slit_width_g140l = float(aperture_stis_g140l.split('X')[1])

aperture_stis_g430m = hdu_stis_g430m[0].header['APERTURE']
slit_length_g430m = float(aperture_stis_g430m.split('X')[0])
slit_width_g430m = float(aperture_stis_g430m.split('X')[1])

ra_stis_g140l = hdu_stis_g140l[1].header["RA_APER"]
dec_stis_g140l = hdu_stis_g140l[1].header["DEC_APER"]
pa_stis_g140l = hdu_stis_g140l[1].header["PA_APER"]

ra_stis_g430m = hdu_stis_g430m[1].header["RA_APER"]
dec_stis_g430m = hdu_stis_g430m[1].header["DEC_APER"]
pa_stis_g430m = hdu_stis_g430m[1].header["PA_APER"]


coords_slit_g140l_world_1 = SkyCoord(ra=hdu_align_g140l[1].header['RA_APER']*u.deg,
                                     dec=hdu_align_g140l[1].header['DEC_APER']*u.deg)
coords_slit_g140l_world_2 = SkyCoord(ra=hdu_align_g140l[4].header['RA_APER']*u.deg,
                                     dec=hdu_align_g140l[4].header['DEC_APER']*u.deg)
coords_slit_g140l_world_3 = SkyCoord(ra=hdu_align_g140l[7].header['RA_APER']*u.deg,
                                     dec=hdu_align_g140l[7].header['DEC_APER']*u.deg)
coords_slit_g140l_pix_1 = wcs_g140l_1.world_to_pixel(coords_slit_g140l_world_1)
coords_slit_g140l_pix_2 = wcs_g140l_2.world_to_pixel(coords_slit_g140l_world_2)
coords_slit_g140l_pix_3 = wcs_g140l_3.world_to_pixel(coords_slit_g140l_world_3)

slit_length_g140l_pix_1 = float(helper_func.transform_world2pix_scale(length_in_arcsec=slit_length_g140l, wcs=wcs_g140l_1))
slit_width_g140l_pix_1 = float(helper_func.transform_world2pix_scale(length_in_arcsec=slit_width_g140l, wcs=wcs_g140l_1))
slit_length_g140l_pix_2 = float(helper_func.transform_world2pix_scale(length_in_arcsec=slit_length_g140l, wcs=wcs_g140l_2))
slit_width_g140l_pix_2 = float(helper_func.transform_world2pix_scale(length_in_arcsec=slit_width_g140l, wcs=wcs_g140l_2))
slit_length_g140l_pix_3 = float(helper_func.transform_world2pix_scale(length_in_arcsec=slit_length_g140l, wcs=wcs_g140l_3))
slit_width_g140l_pix_3 = float(helper_func.transform_world2pix_scale(length_in_arcsec=slit_width_g140l, wcs=wcs_g140l_3))

coords_slit_g430m_world_1 = SkyCoord(ra=hdu_align_g430m[1].header['RA_APER']*u.deg,
                                     dec=hdu_align_g430m[1].header['DEC_APER']*u.deg)
coords_slit_g430m_world_2 = SkyCoord(ra=hdu_align_g430m[4].header['RA_APER']*u.deg,
                                     dec=hdu_align_g430m[4].header['DEC_APER']*u.deg)
coords_slit_g430m_world_3 = SkyCoord(ra=hdu_align_g430m[7].header['RA_APER']*u.deg,
                                     dec=hdu_align_g430m[7].header['DEC_APER']*u.deg)
coords_slit_g430m_pix_1 = wcs_g430m_1.world_to_pixel(coords_slit_g430m_world_1)
coords_slit_g430m_pix_2 = wcs_g430m_2.world_to_pixel(coords_slit_g430m_world_2)
coords_slit_g430m_pix_3 = wcs_g430m_3.world_to_pixel(coords_slit_g430m_world_3)

slit_length_g430m_pix_1 = float(helper_func.transform_world2pix_scale(length_in_arcsec=slit_length_g430m, wcs=wcs_g430m_1))
slit_width_g430m_pix_1 = float(helper_func.transform_world2pix_scale(length_in_arcsec=slit_width_g430m, wcs=wcs_g430m_1))
slit_length_g430m_pix_2 = float(helper_func.transform_world2pix_scale(length_in_arcsec=slit_length_g430m, wcs=wcs_g430m_2))
slit_width_g430m_pix_2 = float(helper_func.transform_world2pix_scale(length_in_arcsec=slit_width_g430m, wcs=wcs_g430m_2))
slit_length_g430m_pix_3 = float(helper_func.transform_world2pix_scale(length_in_arcsec=slit_length_g430m, wcs=wcs_g430m_3))
slit_width_g430m_pix_3 = float(helper_func.transform_world2pix_scale(length_in_arcsec=slit_width_g430m, wcs=wcs_g430m_3))


figure = plt.figure(figsize=(20, 15))
fontsize = 23
ax_hst_g140l_1 = figure.add_axes([0.05, 0.54, 0.3, 0.45], projection=wcs_g140l_1)
ax_hst_g140l_2 = figure.add_axes([0.37, 0.54, 0.3, 0.45], projection=wcs_g140l_2)
ax_hst_g140l_3 = figure.add_axes([0.69, 0.54, 0.3, 0.45], projection=wcs_g140l_3)

ax_hst_g430m_1 = figure.add_axes([0.05, 0.04, 0.3, 0.45], projection=wcs_g430m_1)
ax_hst_g430m_2 = figure.add_axes([0.37, 0.04, 0.3, 0.45], projection=wcs_g430m_2)
ax_hst_g430m_3 = figure.add_axes([0.69, 0.04, 0.3, 0.45], projection=wcs_g430m_3)

ax_hst_g140l_1.imshow(hdu_align_g140l[1].data, origin='lower')
ax_hst_g140l_2.imshow(hdu_align_g140l[4].data, origin='lower')
ax_hst_g140l_3.imshow(hdu_align_g140l[7].data, origin='lower')

ax_hst_g430m_1.imshow(hdu_align_g430m[1].data, origin='lower')
ax_hst_g430m_2.imshow(hdu_align_g430m[4].data, origin='lower')
ax_hst_g430m_3.imshow(hdu_align_g430m[7].data, origin='lower')


plot_slit(ax=ax_hst_g140l_1, coords_slit_pix=coords_slit_g140l_pix_1, slit_length=slit_length_g140l_pix_1,
          slit_width=slit_width_g140l_pix_1, slit_pa=(pa_stis_g140l+90))
plot_slit(ax=ax_hst_g140l_2, coords_slit_pix=coords_slit_g140l_pix_2, slit_length=slit_length_g140l_pix_2,
          slit_width=slit_width_g140l_pix_2, slit_pa=(pa_stis_g140l+90))
plot_slit(ax=ax_hst_g140l_3, coords_slit_pix=coords_slit_g140l_pix_3, slit_length=slit_length_g140l_pix_3,
          slit_width=slit_width_g140l_pix_3, slit_pa=(pa_stis_g140l+90))

plot_slit(ax=ax_hst_g430m_1, coords_slit_pix=coords_slit_g430m_pix_1, slit_length=slit_length_g430m_pix_1,
          slit_width=slit_width_g430m_pix_1, slit_pa=(pa_stis_g430m+90))
plot_slit(ax=ax_hst_g430m_2, coords_slit_pix=coords_slit_g430m_pix_2, slit_length=slit_length_g430m_pix_2,
          slit_width=slit_width_g430m_pix_2, slit_pa=(pa_stis_g430m+90))
plot_slit(ax=ax_hst_g430m_3, coords_slit_pix=coords_slit_g430m_pix_3, slit_length=slit_length_g430m_pix_3,
          slit_width=slit_width_g430m_pix_3, slit_pa=(pa_stis_g430m+90))


# get offset from the brightest spot
peak_coords_pix_g140l, peak_table = analysis_tools.AnalysisTools.sep_find_peak_in_rad(data=hdu_align_g140l[7].data,
                                                                                      err=None,
                                                                                      pixel_coordinates=coords_slit_g140l_pix_3,
                                                                                      pix_radius=10)
print('peak_coords_pix_g140l ', peak_coords_pix_g140l)
peak_coords_world_g140l = wcs_g140l_3.pixel_to_world(peak_coords_pix_g140l[0][0], peak_coords_pix_g140l[1][0])
print('peak_coords_world_g140l ', peak_coords_world_g140l)
ax_hst_g140l_3.scatter(peak_coords_pix_g140l[0], peak_coords_pix_g140l[1], c='g', s=120)
ra_offset_g140l = peak_coords_world_g140l.ra - coords_slit_g140l_world_3.ra
dec_offset_g140l = peak_coords_world_g140l.dec - coords_slit_g140l_world_3.dec
print('ra_offset_g140l ', ra_offset_g140l)
print('dec_offset_g140l ', dec_offset_g140l)


# get offset from the brightest spot
peak_coords_pix_g430m, peak_table = analysis_tools.AnalysisTools.sep_find_peak_in_rad(data=hdu_align_g430m[7].data,
                                                                                      err=None,
                                                                                      pixel_coordinates=coords_slit_g430m_pix_3,
                                                                                      pix_radius=10)
print('peak_coords_pix_g430m ', peak_coords_pix_g430m)
peak_coords_world_g430m = wcs_g430m_3.pixel_to_world(peak_coords_pix_g430m[0][0], peak_coords_pix_g430m[1][0])
print('peak_coords_world_g430m ', peak_coords_world_g430m)
ax_hst_g430m_3.scatter(peak_coords_pix_g430m[0], peak_coords_pix_g430m[1], c='g', s=120)
ra_offset_g430m = peak_coords_world_g430m.ra - coords_slit_g430m_world_3.ra
dec_offset_g430m = peak_coords_world_g430m.dec - coords_slit_g430m_world_3.dec
print('ra_offset_g430m ', ra_offset_g430m)
print('dec_offset_g430m ', dec_offset_g430m)

ax_hst_g140l_1.axis('off')
ax_hst_g140l_2.axis('off')
ax_hst_g140l_3.axis('off')
ax_hst_g430m_1.axis('off')
ax_hst_g430m_2.axis('off')
ax_hst_g430m_3.axis('off')

ax_hst_g140l_1.set_title('G140L first pointing', fontsize=fontsize)
ax_hst_g140l_2.set_title('G140L tuned to source', fontsize=fontsize)
ax_hst_g140l_3.set_title('G140L fine tuning', fontsize=fontsize)
ax_hst_g430m_1.set_title('G430M first pointing', fontsize=fontsize)
ax_hst_g430m_2.set_title('G430M tuned to source', fontsize=fontsize)
ax_hst_g430m_3.set_title('G430M fine tuning', fontsize=fontsize)


plt.savefig('plot_output/slit_calibration_%s.png' % target)

plt.close()
plt.cla()


# plotting of slit_arrangement

hdu_img = fits.open(file_name_dict[target]['file_name_img'])
wcs_img = wcs.WCS(hdu_img['SCI'].header)
img_data = hdu_img['SCI'].data

# get image cutout

cutout_size = (15, 22.5)

coords_img_world = SkyCoord(file_name_dict[target]['coords_img'], unit=(u.hourangle, u.deg))
coords_slit_center_world = SkyCoord(file_name_dict[target]['coords_slit_center'], unit=(u.hourangle, u.deg))
# coords_slit_center_world_b = SkyCoord(file_name_dict[target]['coords_source_b'], unit=(u.hourangle, u.deg))


cutout_img_large = helper_func.get_img_cutout(img=img_data, wcs=wcs_img, coord=coords_img_world, cutout_size=cutout_size)
cutout_img_zoom = helper_func.get_img_cutout(img=img_data, wcs=wcs_img, coord=coords_slit_center_world, cutout_size=(3, 3))
# cutout_img_zoom_b = helper_func.get_img_cutout(img=img_data, wcs=wcs_img, coord=coords_slit_center_world_b, cutout_size=(3, 3))


coords_slit_center_pix = cutout_img_zoom.wcs.world_to_pixel(coords_slit_center_world)
# coords_slit_center_b_pix = cutout_img_zoom.wcs.world_to_pixel(coords_slit_center_world_b)


# get offset from the brightest spot
peak_coords_zoom_pix, peak_table = analysis_tools.AnalysisTools.sep_find_peak_in_rad(data=cutout_img_zoom.data,
                                                                                      err=None,
                                                                                      pixel_coordinates=coords_slit_center_pix,
                                                                                      pix_radius=10)


peak_coords_zoom_world = cutout_img_zoom.wcs.pixel_to_world(peak_coords_zoom_pix[0], peak_coords_zoom_pix[1])

slit_pos_g140l_world = SkyCoord(ra=peak_coords_zoom_world.ra - ra_offset_g140l,
                                dec=peak_coords_zoom_world.dec - dec_offset_g140l)
slit_pos_g430m_world = SkyCoord(ra=peak_coords_zoom_world.ra - ra_offset_g430m,
                                dec=peak_coords_zoom_world.dec - dec_offset_g430m)
#
# slit_pos_g140l_world = SkyCoord(ra=peak_coords_zoom_world.ra,
#                                 dec=peak_coords_zoom_world.dec)
# slit_pos_g430m_world = SkyCoord(ra=peak_coords_zoom_world.ra,
#                                 dec=peak_coords_zoom_world.dec)

slit_pos_g140l_pix_large = cutout_img_large.wcs.world_to_pixel(slit_pos_g140l_world)
slit_pos_g430m_pix_large = cutout_img_large.wcs.world_to_pixel(slit_pos_g430m_world)
slit_pos_g140l_pix_zoom = cutout_img_zoom.wcs.world_to_pixel(slit_pos_g140l_world)
slit_pos_g430m_pix_zoom = cutout_img_zoom.wcs.world_to_pixel(slit_pos_g430m_world)
# slit_pos_g140l_pix_zoom_b = cutout_img_zoom_b.wcs.world_to_pixel(slit_pos_g140l_world)
# slit_pos_g430m_pix_zoom_b = cutout_img_zoom_b.wcs.world_to_pixel(slit_pos_g430m_world)

figure = plt.figure(figsize=(20, 14))
fontsize = 23
ax_hst_img = figure.add_axes([0.04, 0.04, 0.95, 0.95], projection=cutout_img_large.wcs)
# ax_hst_zoom = figure.add_axes([0.55, 0.10, 0.4, 0.4], projection=cutout_img_zoom.wcs)
# ax_hst_zoom = figure.add_axes([0.15, 0.55, 0.4, 0.4], projection=cutout_img_zoom.wcs)
ax_hst_zoom = figure.add_axes([0.64, 0.5, 0.4, 0.4], projection=cutout_img_zoom.wcs)

plot_slit(ax=ax_hst_img, coords_slit_pix=slit_pos_g140l_pix_large, slit_length=slit_length_g140l_pix_1,
          slit_width=slit_width_g140l_pix_1, slit_pa=(pa_stis_g140l + 90), color='green', lw=5, linestyle='-', plot_scatter=False)
plot_slit(ax=ax_hst_img, coords_slit_pix=slit_pos_g430m_pix_large, slit_length=slit_length_g430m_pix_1,
          slit_width=slit_width_g430m_pix_1, slit_pa=(pa_stis_g430m + 90), color='blue', lw=2, linestyle='--', plot_scatter=False)

plot_slit(ax=ax_hst_zoom, coords_slit_pix=slit_pos_g140l_pix_zoom, slit_length=slit_length_g140l_pix_1,
          slit_width=slit_width_g140l_pix_1, slit_pa=(pa_stis_g140l + 90), color='green', lw=5, linestyle='-', plot_scatter=False)
plot_slit(ax=ax_hst_zoom, coords_slit_pix=slit_pos_g430m_pix_zoom, slit_length=slit_length_g430m_pix_1,
          slit_width=slit_width_g430m_pix_1, slit_pa=(pa_stis_g430m + 90), color='blue', lw=2, linestyle='--', plot_scatter=False)

# plot_slit(ax=ax_hst_zoom_b, coords_slit_pix=slit_pos_g140l_pix_zoom_b, slit_length=slit_length_g140l_pix_1,
#           slit_width=slit_width_g140l_pix_1, slit_pa=(pa_stis_g140l + 90), color='green', lw=5, linestyle='-', plot_scatter=False)
# plot_slit(ax=ax_hst_zoom_b, coords_slit_pix=slit_pos_g430m_pix_zoom_b, slit_length=slit_length_g430m_pix_1,
#           slit_width=slit_width_g430m_pix_1, slit_pa=(pa_stis_g430m + 90), color='blue', lw=2, linestyle='--', plot_scatter=False)

ax_hst_img.plot([], [], color='green', linewidth=5, linestyle='-', label='G140L')
ax_hst_img.plot([], [], color='blue', linewidth=2, linestyle='--', label='G430M')
ax_hst_img.legend(frameon=False, loc=2, labelcolor='white', fontsize=fontsize)


# plot circles
coords_source_a_world = SkyCoord(file_name_dict[target]['coords_source_a'], unit=(u.hourangle, u.deg))
# coords_source_b_world = SkyCoord(file_name_dict[target]['coords_source_b'], unit=(u.hourangle, u.deg))
coords_source_a_pix = cutout_img_zoom.wcs.world_to_pixel(coords_source_a_world)
# coords_source_b_pix = cutout_img_zoom_b.wcs.world_to_pixel(coords_source_b_world)
plotting_tools.plot_coord_circle(ax=ax_hst_zoom, pos=coords_source_a_world, rad=0.4, color='white',
                                 linestyle='-', linewidth=2, alpha=1., fill=False)

# plotting_tools.plot_coord_circle(ax=ax_hst_zoom_b, pos=coords_source_b_world, rad=0.4, color='white',
#                                  linestyle='-', linewidth=2, alpha=1., fill=False)

ax_hst_zoom.text(coords_source_a_pix[0] - 4, coords_source_a_pix[1] + 7, 'A',
                horizontalalignment='right', verticalalignment='bottom', color='white', fontsize=fontsize)
# ax_hst_zoom_b.text(coords_source_b_pix[0] - 4, coords_source_b_pix[1] + 7, 'B',
#                 horizontalalignment='right', verticalalignment='bottom', color='white', fontsize=fontsize)
#

mean, median, std = sigma_clipped_stats(cutout_img_large.data, sigma=3.0)
# ax_hst_img.imshow(cutout_img_large.data, vmin=mean-std, vmax=mean+50*std, cmap='inferno')
# ax_hst_zoom.imshow(cutout_img_zoom.data, vmin=mean-std, vmax=mean+100*std, cmap='inferno')
print('mean, median, std ', mean, median, std)
display_data_img = cutout_img_large.data
display_data_zoom = cutout_img_zoom.data
# display_data_zoom_b = cutout_img_zoom_b.data
display_data_img[((display_data_img <= 0) | np.isnan(display_data_img))] = mean + std
display_data_zoom[((display_data_zoom <= 0) | np.isnan(display_data_zoom))] = mean + std
# display_data_zoom_b[((display_data_zoom_b <= 0) | np.isnan(display_data_zoom_b))] = mean + std
ax_hst_img.imshow(np.log10(display_data_img), vmin=-3, vmax=1, cmap='inferno')
ax_hst_zoom.imshow(np.log10(display_data_zoom), vmin=-3, vmax=1, cmap='inferno')
# ax_hst_zoom_b.imshow(np.log10(display_data_zoom_b), vmin=-3, vmax=1, cmap='inferno')



ax_hst_img.set_title(target.upper(), fontsize=fontsize)


ax_hst_img.text(cutout_img_large.data.shape[0] * 0.05, cutout_img_large.data.shape[1] * 0.05,
                file_name_dict[target]['filter_name_img'],
                horizontalalignment='left', verticalalignment='bottom', color='white', fontsize=fontsize)


plotting_tools.arr_axis_params(ax=ax_hst_img, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label='R.A. (2000.0)', dec_axis_label='DEC. (2000.0)',
                               ra_minpad=0.3, dec_minpad=0.8, tick_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=5)

plotting_tools.arr_axis_params(ax=ax_hst_zoom, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label=' ', dec_axis_label=' ',
                               ra_minpad=0.3, dec_minpad=0.8, tick_color='white', label_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=4, dec_tick_num=4)

# plotting_tools.arr_axis_params(ax=ax_hst_zoom_b, ra_tick_label=True, dec_tick_label=True,
#                                ra_axis_label=' ', dec_axis_label=' ',
#                                ra_minpad=0.3, dec_minpad=0.8, tick_color='white', label_color='white',
#                                fontsize=fontsize, labelsize=fontsize, ra_tick_num=4, dec_tick_num=4)

# plt.show()
plt.savefig('plot_output/alignment_%s.png' % target)
plt.savefig('plot_output/alignment_%s.pdf' % target)

exit()
