import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm
import astropy.units as u
from astropy.coordinates import SkyCoord
from photometry_tools import helper_func, plotting_tools, analysis_tools
from regions import PixCoord, RectanglePixelRegion
from astropy.stats import sigma_clipped_stats

from astropy.wcs import WCS
from astropy.visualization.wcsaxes import SphericalCircle
import multicolorfits as mcf
from astropy.wcs.utils import fit_wcs_from_points


hdu_hst_r = fits.open('/home/benutzer/data/observation/heII_paper/ngc3049/img/o54a01s6q_flt.fits')
hdu_hst_g = fits.open('/home/benutzer/data/observation/heII_paper/ngc3049/img/o54a01010_flt.fits')
data_r = hdu_hst_r[1].data
data_g = hdu_hst_g[1].data
wcs_r = WCS(hdu_hst_r[1].header)
wcs_g = WCS(hdu_hst_g[1].header)
coord_dict = {
'coords_img': '9h54m49.6s 9d16m17.0s',
'coords_slit_center': '9h54m49.56s 9d16m17.4s',
'coords_source_a': '9h54m49.50s 9d16m16.5s',
'coords_source_b': '9h54m49.613s 9d16m18.07s',
}

# coords_cent = SkyCoord(coord_dict['coords_img'], unit=(u.hourangle, u.deg))
# cutout_r = helper_func.get_img_cutout(img=data_r, wcs=wcs_r, coord=coords_cent, cutout_size=(20, 20*1.5))
# cutout_g = helper_func.get_img_cutout(img=data_g, wcs=wcs_g, coord=coords_cent, cutout_size=(20, 20*1.5))

# # # update WCS
# stars = [np.array([579.89, 586.05, 549.80, 393.50]), np.array([470.46, 514.07, 377.98, 78.70])]
# known_coords = SkyCoord(['9h54m49.51s 9d16m16.0s', '9h54m49.454s 9d16m15.4s', '9h54m49.630s 9d16m17.7s',
#                          '9h54m49.970s 9d16m24.3s'], unit=(u.hourangle, u.deg))
# new_cutout_r = fit_wcs_from_points(xy=stars, world_coords=known_coords, projection=wcs_r)

stars = [np.array([579.89, 586.05, 549.80]), np.array([470.46, 514.07, 377.98])]
known_coords = SkyCoord(['9h54m49.51s 9d16m16.0s', '9h54m49.454s 9d16m15.4s', '9h54m49.630s 9d16m17.7s'], unit=(u.hourangle, u.deg))
new_cutout_r = fit_wcs_from_points(xy=stars, world_coords=known_coords, projection=wcs_r)

stars = [np.array([579.89, 586.05, 549.80]), np.array([470.46, 514.07, 377.98])]
known_coords = SkyCoord(['9h54m49.51s 9d16m16.0s', '9h54m49.454s 9d16m15.4s', '9h54m49.630s 9d16m17.7s'], unit=(u.hourangle, u.deg))
new_cutout_r = fit_wcs_from_points(xy=stars, world_coords=known_coords, projection=wcs_r)



figure = plt.figure(figsize=(20, 10))
fontsize = 35
ax_img_r = figure.add_axes([0.05, 0.05, 0.52, 0.52], projection=new_cutout_r)
ax_img_g = figure.add_axes([0.53, 0.05, 0.52, 0.52], projection=wcs_g)

ax_img_r.imshow(np.log10(data_r), origin='lower')
ax_img_g.imshow(np.log10(data_g))

plt.show()

exit()

# <
#     # # reproject hst_cutouts to b
#     new_wcs = cutout_b.wcs
#     new_shape = cutout_b.data.shape
#     hst_img_g = plotting_tools.reproject_image(data=cutout_g.data,
#                                                wcs=cutout_g.wcs,
#                                                new_wcs=new_wcs,
#                                                new_shape=new_shape)
#     hst_img_r = plotting_tools.reproject_image(data=cutout_r.data,
#                                                wcs=cutout_r.wcs,
#                                                new_wcs=new_wcs,
#                                                new_shape=new_shape)
#
#     grey_hst_r = mcf.greyRGBize_image(hst_img_r, rescalefn='asinh', scaletype='perc', min_max=[1.0, 99.9],
#                                       gamma=file_name_dict[target]['gamma_b'], checkscale=False)
#     grey_hst_g = mcf.greyRGBize_image(hst_img_g, rescalefn='asinh', scaletype='perc', min_max=[1.0, 99.9],
#                                       gamma=file_name_dict[target]['gamma_g'], checkscale=False)
#     grey_hst_b = mcf.greyRGBize_image(cutout_b.data, rescalefn='asinh', scaletype='perc', min_max=[1.0, 99.9],
#                                       gamma=file_name_dict[target]['gamma_r'], checkscale=False)
#     hst_r_purple = mcf.colorize_image(grey_hst_r, file_name_dict[target]['color_r'], colorintype='hex', gammacorr_color=2.2)
#     hst_g_orange = mcf.colorize_image(grey_hst_g, file_name_dict[target]['color_g'], colorintype='hex', gammacorr_color=2.2)
#     hst_b_blue = mcf.colorize_image(grey_hst_b, file_name_dict[target]['color_b'], colorintype='hex', gammacorr_color=2.2)
#     rgb_hst_image = mcf.combine_multicolor([hst_r_purple, hst_g_orange, hst_b_blue],
#                                            gamma=file_name_dict[target]['gamma_rgb'], inverse=False)
#
#     return rgb_hst_image, new_wcs


def get_slit_pos(target):

    # load slit_observations
    hdu_stis_g140l = fits.open(file_name_dict[target]['file_name_stis_g140l'])
    hdu_align_g140l = fits.open(file_name_dict[target]['file_name_align_g140l'])

    hdu_stis_g430m = fits.open(file_name_dict[target]['file_name_stis_g430m'])
    hdu_align_g430m = fits.open(file_name_dict[target]['file_name_align_g430m'])

    wcs_g140l_1 = WCS(hdu_align_g140l[1].header)
    wcs_g140l_3 = WCS(hdu_align_g140l[7].header)

    wcs_g430m_1 = WCS(hdu_align_g430m[1].header)
    wcs_g430m_3 = WCS(hdu_align_g430m[7].header)

    aperture_stis_g140l = hdu_stis_g140l[0].header['APERTURE']
    slit_length_g140l = float(aperture_stis_g140l.split('X')[0])
    slit_width_g140l = float(aperture_stis_g140l.split('X')[1])

    aperture_stis_g430m = hdu_stis_g430m[0].header['APERTURE']
    slit_length_g430m = float(aperture_stis_g430m.split('X')[0])
    slit_width_g430m = float(aperture_stis_g430m.split('X')[1])

    pa_stis_g140l = hdu_stis_g140l[1].header["PA_APER"]

    pa_stis_g430m = hdu_stis_g430m[1].header["PA_APER"]

    coords_slit_g140l_world_3 = SkyCoord(ra=hdu_align_g140l[7].header['RA_APER']*u.deg,
                                         dec=hdu_align_g140l[7].header['DEC_APER']*u.deg)
    coords_slit_g140l_pix_3 = wcs_g140l_3.world_to_pixel(coords_slit_g140l_world_3)

    slit_length_g140l_pix_1 = float(helper_func.transform_world2pix_scale(length_in_arcsec=slit_length_g140l, wcs=wcs_g140l_1))
    slit_width_g140l_pix_1 = float(helper_func.transform_world2pix_scale(length_in_arcsec=slit_width_g140l, wcs=wcs_g140l_1))

    coords_slit_g430m_world_3 = SkyCoord(ra=hdu_align_g430m[7].header['RA_APER']*u.deg,
                                         dec=hdu_align_g430m[7].header['DEC_APER']*u.deg)
    coords_slit_g430m_pix_3 = wcs_g430m_3.world_to_pixel(coords_slit_g430m_world_3)

    slit_length_g430m_pix_1 = float(helper_func.transform_world2pix_scale(length_in_arcsec=slit_length_g430m, wcs=wcs_g430m_1))
    slit_width_g430m_pix_1 = float(helper_func.transform_world2pix_scale(length_in_arcsec=slit_width_g430m, wcs=wcs_g430m_1))

    # get offset from the brightest spot
    peak_coords_pix_g140l, peak_table = analysis_tools.AnalysisTools.sep_find_peak_in_rad(data=hdu_align_g140l[7].data,
                                                                                          err=None,
                                                                                          pixel_coordinates=coords_slit_g140l_pix_3,
                                                                                          pix_radius=10)
    peak_coords_world_g140l = wcs_g140l_3.pixel_to_world(peak_coords_pix_g140l[0][0], peak_coords_pix_g140l[1][0])
    ra_offset_g140l = peak_coords_world_g140l.ra - coords_slit_g140l_world_3.ra
    dec_offset_g140l = peak_coords_world_g140l.dec - coords_slit_g140l_world_3.dec

    # get offset from the brightest spot
    peak_coords_pix_g430m, peak_table = analysis_tools.AnalysisTools.sep_find_peak_in_rad(data=hdu_align_g430m[7].data,
                                                                                          err=None,
                                                                                          pixel_coordinates=coords_slit_g430m_pix_3,
                                                                                          pix_radius=10)
    peak_coords_world_g430m = wcs_g430m_3.pixel_to_world(peak_coords_pix_g430m[0][0], peak_coords_pix_g430m[1][0])
    ra_offset_g430m = peak_coords_world_g430m.ra - coords_slit_g430m_world_3.ra
    dec_offset_g430m = peak_coords_world_g430m.dec - coords_slit_g430m_world_3.dec

    slit_dict = {
        'ra_offset_g140l': ra_offset_g140l,
        'dec_offset_g140l': dec_offset_g140l,
        'dec_offset_g430m': dec_offset_g430m,
        'ra_offset_g430m': ra_offset_g430m,
        'pa_stis_g140l': pa_stis_g140l,
        'pa_stis_g430m': pa_stis_g430m,
        'slit_length_g140l_pix_1': slit_length_g140l_pix_1,
        'slit_width_g140l_pix_1': slit_width_g140l_pix_1,
        'slit_length_g430m_pix_1': slit_length_g430m_pix_1,
        'slit_width_g430m_pix_1': slit_width_g430m_pix_1
    }
    return slit_dict


def display_slit(ax, target, wcs, data, slit_dict):

    ra_offset_g140l = slit_dict['ra_offset_g140l']
    dec_offset_g140l = slit_dict['dec_offset_g140l']
    dec_offset_g430m = slit_dict['dec_offset_g430m']
    ra_offset_g430m = slit_dict['ra_offset_g430m']
    pa_stis_g140l = slit_dict['pa_stis_g140l']
    pa_stis_g430m = slit_dict['pa_stis_g430m']
    slit_length_g140l_pix_1 = slit_dict['slit_length_g140l_pix_1']
    slit_width_g140l_pix_1 = slit_dict['slit_width_g140l_pix_1']
    slit_length_g430m_pix_1 = slit_dict['slit_length_g430m_pix_1']
    slit_width_g430m_pix_1 = slit_dict['slit_width_g430m_pix_1']



    coords_slit_center_world = SkyCoord(file_name_dict[target]['coords_slit_center'], unit=(u.hourangle, u.deg))

    coords_slit_center_pix = wcs.world_to_pixel(coords_slit_center_world)

    # get offset from the brightest spot
    peak_coords_pix, peak_table = analysis_tools.AnalysisTools.sep_find_peak_in_rad(data=data,
                                                                                    err=None,
                                                                                    pixel_coordinates=coords_slit_center_pix,
                                                                                    pix_radius=10)

    peak_coords_world = wcs.pixel_to_world(peak_coords_pix[0], peak_coords_pix[1])

    slit_pos_g140l_world = SkyCoord(ra=peak_coords_world.ra - ra_offset_g140l,
                                    dec=peak_coords_world.dec - dec_offset_g140l)
    slit_pos_g430m_world = SkyCoord(ra=peak_coords_world.ra - dec_offset_g430m,
                                    dec=peak_coords_world.dec - ra_offset_g430m)

    slit_pos_g140l_pix = wcs.world_to_pixel(slit_pos_g140l_world)
    slit_pos_g430m_pix = wcs.world_to_pixel(slit_pos_g430m_world)

    plot_slit(ax=ax, coords_slit_pix=slit_pos_g140l_pix, slit_length=slit_length_g140l_pix_1,
              slit_width=slit_width_g140l_pix_1, slit_pa=(pa_stis_g140l + 90), color='green', lw=5,
              linestyle='-', plot_scatter=False)
    plot_slit(ax=ax, coords_slit_pix=slit_pos_g430m_pix, slit_length=slit_length_g430m_pix_1,
              slit_width=slit_width_g430m_pix_1, slit_pa=(pa_stis_g430m + 90), color='blue', lw=2,
              linestyle='--', plot_scatter=False)


def display_circ(ax, wcs, coords, text='A', x_offset=-5, y_offset=7, fontsize=25):
    coords_source_a_world = SkyCoord(coords, unit=(u.hourangle, u.deg))
    coords_source_a_pix = wcs.world_to_pixel(coords_source_a_world)
    plotting_tools.plot_coord_circle(ax=ax, pos=coords_source_a_world, rad=0.2, color='white',
                                     linestyle='-', linewidth=2, alpha=1., fill=False)
    ax.text(coords_source_a_pix[0] + x_offset, coords_source_a_pix[1] + y_offset, text,
                    horizontalalignment='right', verticalalignment='bottom', color='white', fontsize=fontsize)


def display_bands(ax, target, target_name, data, fontsize):


    ax.text(data.shape[1] * 0.03, data.shape[0] * 0.97,
            target_name,
            horizontalalignment='left', verticalalignment='top', color='white', fontsize=fontsize)
    ax.text(data.shape[1] * 0.03, data.shape[0] * 0.92,
            file_name_dict[target]['filter_name_r'],
            horizontalalignment='left', verticalalignment='top',
            color=file_name_dict[target]['color_r'], fontsize=fontsize)
    if 'filter_name_g' in file_name_dict[target].keys():
        ax.text(data.shape[1] * 0.03, data.shape[0] * 0.87,
                file_name_dict[target]['filter_name_g'],
                horizontalalignment='left', verticalalignment='top',
                color=file_name_dict[target]['color_g'], fontsize=fontsize)
    if 'filter_name_b' in file_name_dict[target].keys():
        ax.text(data.shape[1] * 0.03, data.shape[0] * 0.82,
                file_name_dict[target]['filter_name_b'],
                horizontalalignment='left', verticalalignment='top',
                color=file_name_dict[target]['color_b'], fontsize=fontsize)


file_name_dict = {
    'he2_10': {

        'file_name_b': '/home/benutzer/data/observation/heII_paper/he2_10/img/hst_06580_02_wfpc2_f555w_pc_drz.fits',
        'file_name_g': '/home/benutzer/data/observation/heII_paper/he2_10/img/hst_11146_01_wfpc2_f814w_wf_drz.fits',
        'file_name_r': '/home/benutzer/data/observation/heII_paper/he2_10/img/hst_06580_02_wfpc2_f658n_pc_drz.fits',
        'filter_name_b': 'F555W',
        'filter_name_g': 'F814W',
        'filter_name_r': 'F658N',
        'gamma_b': 2.8,
        'gamma_g': 2.8,
        'gamma_r': 2.8,
        'gamma_rgb': 4.0,
        'color_b': '#0096FF',
        'color_g': '#4CBB17',
        'color_r': '#FF4433',
        'file_name_align_g140l': '/home/benutzer/data/observation/heII_paper/he2_10/stis/O6ER02BYQ/o6er02byq_raw.fits',
        'file_name_stis_g140l': '/home/benutzer/data/observation/heII_paper/he2_10/stis/O6ER02020/o6er02020_flt.fits',
        'file_name_align_g430m': '/home/benutzer/data/observation/heII_paper/he2_10/stis/OE0701GYQ/oe0701gyq_raw.fits',
        'file_name_stis_g430m': '/home/benutzer/data/observation/heII_paper/he2_10/stis/OE0701020/oe0701020_flt.fits',
        # 'coords_img': '8h36m15.095s -26d24m32.5s',
        'coords_img': '8h36m15.199s -26d24m35.30s',
        'coords_slit_center': '8h36m15.199s -26d24m33.68s',
        'coords_zoom_position': '8h36m15.166s -26d24m33.75s',
        'coords_source_a': '8h36m15.199s -26d24m33.68s',
        'coords_source_b': '8h36m15.166s -26d24m33.75s',
        'coords_source_c': '8h36m15.136s -26d24m33.78s',
        'coords_source_d': '8h36m15.114s -26d24m33.84s',
    },
    'mrk33': {
        'file_name_b': '/home/benutzer/data/observation/heII_paper/mrk33/img/hst_11987_04_wfpc2_f555w_wf_drz.fits',
        'file_name_g': '/home/benutzer/data/observation/heII_paper/mrk33/img/hst_11987_04_wfpc2_f814w_wf_drz.fits',
        'file_name_r': '/home/benutzer/data/observation/heII_paper/mrk33/img/hst_11987_04_wfpc2_f658n_wf_drz.fits',
        'filter_name_b': 'F555W',
        'filter_name_g': 'F814W',
        'filter_name_r': 'F658N',
        'gamma_b': 2.8,
        'gamma_g': 2.8,
        'gamma_r': 2.8,
        'gamma_rgb': 4.0,
        'color_b': '#0096FF',
        'color_g': '#4CBB17',
        'color_r': '#FF4433',


        'file_name_align_g140l': '/home/benutzer/data/observation/heII_paper/mrk33/stsi/O6ER04Y6Q/o6er04y6q_raw.fits',
        'file_name_stis_g140l': '/home/benutzer/data/observation/heII_paper/mrk33/stsi/O6ER04020/o6er04020_flt.fits',
        'file_name_align_g430m': '/home/benutzer/data/observation/heII_paper/mrk33/stsi/OE0704OBQ/oe0704obq_raw.fits',
        'file_name_stis_g430m': '/home/benutzer/data/observation/heII_paper/mrk33/stsi/OE0704020/oe0704020_flt.fits',
        'coords_img': '10h32m31.28s +54d24m04.2s',
        'coords_slit_center': '10h32m31.91s +54d24m03.1s',
        'coords_source_a': '10h32m31.95s +54d24m02.3s',
        'coords_source_b': '10h32m31.83s +54d24m04.0s',
    },
    # 'ngc_3049': {
    #     'file_name_r': '/home/benutzer/data/observation/heII_paper/ngc3049/img/HST_11080_a1_NIC_NIC3_F190N_drz.fits',
    #     'filter_name_r': 'F190N',
    #     'gamma_r': 1.5,
    #     'gamma_rgb': 1.5,
    #     'color_r': '#FF4433',
    #
    #     'file_name_align_g140l': '/home/benutzer/data/observation/heII_paper/ngc3049/stis/O54A10BHQ/o54a10bhq_raw.fits',
    #     'file_name_stis_g140l': '/home/benutzer/data/observation/heII_paper/ngc3049/stis/O54A10030/o54a10030_flt.fits',
    #     'file_name_align_g430m': '/home/benutzer/data/observation/heII_paper/ngc3049/stis/OE0702Y0Q/oe0702y0q_raw.fits',
    #     'file_name_stis_g430m': '/home/benutzer/data/observation/heII_paper/ngc3049/stis/OE0702020/oe0702020_flt.fits',
    #     'coords_img': '9h54m49.6s 9d16m17.0s',
    #     'coords_slit_center': '9h54m49.56s 9d16m17.4s',
    #     'coords_source_a': '9h54m49.50s 9d16m16.5s',
    #     'coords_source_b': '9h54m49.613s 9d16m18.07s',
    # },

        'ngc_3049': {
        'file_name_r': '/home/benutzer/data/observation/heII_paper/ngc3049/img/o54a01s6q_flt.fits',
        'file_name_g': '/home/benutzer/data/observation/heII_paper/ngc3049/img/o54a01010_flt.fits',
        'file_name_b': '/home/benutzer/data/observation/heII_paper/ngc3049/img/o54a01020_flt.fits',
        'filter_name_r': 'MIRFUV',
        'filter_name_g': 'MIRVIS',
        'filter_name_b': 'MIRVIS',
        'gamma_b': 2.8,
        'gamma_g': 2.8,
        'gamma_r': 2.8,
        'gamma_rgb': 4.0,
        'color_b': '#0096FF',
        'color_g': '#4CBB17',
        'color_r': '#FF4433',

        'file_name_align_g140l': '/home/benutzer/data/observation/heII_paper/ngc3049/stis/O54A10BHQ/o54a10bhq_raw.fits',
        'file_name_stis_g140l': '/home/benutzer/data/observation/heII_paper/ngc3049/stis/O54A10030/o54a10030_flt.fits',
        'file_name_align_g430m': '/home/benutzer/data/observation/heII_paper/ngc3049/stis/OE0702Y0Q/oe0702y0q_raw.fits',
        'file_name_stis_g430m': '/home/benutzer/data/observation/heII_paper/ngc3049/stis/OE0702020/oe0702020_flt.fits',
        'coords_img': '9h54m49.6s 9d16m17.0s',
        'coords_slit_center': '9h54m49.56s 9d16m17.4s',
        'coords_source_a': '9h54m49.50s 9d16m16.5s',
        'coords_source_b': '9h54m49.613s 9d16m18.07s',
    },

    'ngc_3125': {
        'file_name_b': '/home/benutzer/data/observation/heII_paper/ngc3125/img/hst_10400_01_acs_hrc_f555w_drz.fits',
        'file_name_g': '/home/benutzer/data/observation/heII_paper/ngc3125/img/hst_10400_01_acs_hrc_f814w_drz.fits',
        'file_name_r': '/home/benutzer/data/observation/heII_paper/ngc3125/img/hst_10400_01_acs_hrc_f658n_drz.fits',
        'filter_name_b': 'F555W',
        'filter_name_g': 'F814W',
        'filter_name_r': 'F658N',
        'gamma_b': 2.8,
        'gamma_g': 2.8,
        'gamma_r': 2.8,
        'gamma_rgb': 4.0,
        'color_b': '#0096FF',
        'color_g': '#4CBB17',
        'color_r': '#FF4433',

        'file_name_align_g140l': '/home/benutzer/data/observation/heII_paper/ngc3125/stis/O6ER03NMQ/o6er03nmq_raw.fits',
        'file_name_stis_g140l': '/home/benutzer/data/observation/heII_paper/ngc3125/stis/O6ER03020/o6er03020_flt.fits',
        'file_name_align_g430m': '/home/benutzer/data/observation/heII_paper/ngc3125/stis/OE0703SNQ/oe0703snq_raw.fits',
        'file_name_stis_g430m': '/home/benutzer/data/observation/heII_paper/ngc3125/stis/OE0703020/oe0703020_flt.fits',
        'coords_img': '10h06m33.3s -29d56m09.0s',
        'coords_slit_center': '10h06m33.26s -29d56m06.76s',
        'coords_source_a': '10h06m33.263s -29d56m06.76s',
    },
    'ngc_4214': {
        'file_name_b': '/home/benutzer/data/observation/heII_paper/ngc4214/img/hst_11360_72_wfc3_uvis_f438w_drz.fits',
        'file_name_g': '/home/benutzer/data/observation/heII_paper/ngc4214/img/hst_11360_72_wfc3_uvis_f814w_drz.fits',
        'file_name_r': '/home/benutzer/data/observation/heII_paper/ngc4214/img/hst_11360_72_wfc3_uvis_f657n_drz.fits',
        'filter_name_b': 'F438W',
        'filter_name_g': 'F814W',
        'filter_name_r': 'F657N',
        'gamma_b': 2.8,
        'gamma_g': 2.8,
        'gamma_r': 2.8,
        'gamma_rgb': 4.0,
        'color_b': '#0096FF',
        'color_g': '#4CBB17',
        'color_r': '#FF4433',


        'file_name_align_g140l': '/home/benutzer/data/observation/heII_paper/ngc4214/stis/O6ER10DCQ/o6er10dcq_raw.fits',
        'file_name_stis_g140l': '/home/benutzer/data/observation/heII_paper/ngc4214/stis/O6ER10020/o6er10020_flt.fits',
        'file_name_align_g430m': '/home/benutzer/data/observation/heII_paper/ngc4214/stis/OE0705S4Q/oe0705s4q_raw.fits',
        'file_name_stis_g430m': '/home/benutzer/data/observation/heII_paper/ngc4214/stis/OE0705020/oe0705020_flt.fits',
        # 'coords_img': '12h15m39.045s +36d19m25.8s',
        'coords_img': '12h15m39.6s +36d19m34.6s',
        'coords_slit_center': '12h15m39.45s +36d19m34.6s',
        'coords_source_a': '12h15m39.45s +36d19m34.6s',
    },
    'ngc_4670': {
        'file_name_b': '/home/benutzer/data/observation/heII_paper/ngc4670/img/hst_06639_03_wfpc2_f336w_pc_drz.fits',
        'file_name_g': '/home/benutzer/data/observation/heII_paper/ngc4670/img/hst_06639_03_wfpc2_f439w_pc_drz.fits',
        'file_name_r': '/home/benutzer/data/observation/heII_paper/ngc4670/img/hst_06639_03_wfpc2_f814w_pc_drz.fits',
        'filter_name_b': 'F336W',
        'filter_name_g': 'F439W',
        'filter_name_r': 'F814W',
        'gamma_b': 2.8,
        'gamma_g': 2.8,
        'gamma_r': 2.8,
        'gamma_rgb': 4.0,
        'color_b': '#0096FF',
        'color_g': '#4CBB17',
        'color_r': '#FF4433',


        'file_name_align_g140l': '/home/benutzer/data/observation/heII_paper/ngc4670/stis/O6ER13MDQ/o6er13mdq_raw.fits',
        'file_name_stis_g140l': '/home/benutzer/data/observation/heII_paper/ngc4670/stis/O6ER13020/o6er13020_flt.fits',
        'file_name_align_g430m': '/home/benutzer/data/observation/heII_paper/ngc4670/stis/OE0706Y1Q/oe0706y1q_raw.fits',
        'file_name_stis_g430m': '/home/benutzer/data/observation/heII_paper/ngc4670/stis/OE0706020/oe0706020_flt.fits',
        'coords_img': '12h45m16.9s +27d07m31.0s',
        'coords_slit_center': '12h45m17.265s +27d07m32.11s',
        'coords_slit_center_b': '12h45m17.11s +27d07m30.0s',
        'coords_source_a': '12h45m17.265s +27d07m32.11s',
        'coords_source_b': '12h45m17.119s +27d07m30.21s',
        'coords_source_c': '12h45m17.104s +27d07m29.95s',
    },
    'tol89': {
        'file_name_b': '/home/benutzer/data/observation/heII_paper/tol89/img/hst_11987_28_wfpc2_f555w_wf_drz.fits',
        'file_name_g': '/home/benutzer/data/observation/heII_paper/tol89/img/hst_11987_28_wfpc2_f814w_wf_drz.fits',
        'file_name_r': '/home/benutzer/data/observation/heII_paper/tol89/img/hst_11987_28_wfpc2_f658n_wf_drz.fits',
        'filter_name_b': 'F555W',
        'filter_name_g': 'F814W',
        'filter_name_r': 'F658N',
        'gamma_b': 2.8,
        'gamma_g': 2.8,
        'gamma_r': 2.8,
        'gamma_rgb': 4.0,
        'color_b': '#0096FF',
        'color_g': '#4CBB17',
        'color_r': '#FF4433',

        'file_name_align_g140l': '/home/benutzer/data/observation/heII_paper/tol89/stis/O54A20EWQ/o54a20ewq_raw.fits',
        'file_name_stis_g140l': '/home/benutzer/data/observation/heII_paper/tol89/stis/O54A20030/o54a20030_flt.fits',
        'file_name_align_g430m': '/home/benutzer/data/observation/heII_paper/tol89/stis/OE0707Q0Q/oe0707q0q_raw.fits',
        'file_name_stis_g430m': '/home/benutzer/data/observation/heII_paper/tol89/stis/OE0707020/oe0707020_flt.fits',
        'coords_img': '14h01m19.92s -33d04m12.7s',
        'coords_slit_center': '14h01m19.934s -33d04m10.91s',
        'coords_source_a': '14h01m19.934s -33d04m10.91s',
    },

    'tol1924_416': {
        'file_name_b': '/home/benutzer/data/observation/heII_paper/tol1924_416/img/hst_06708_01_wfpc2_f336w_pc_drz.fits',
        'file_name_g': '/home/benutzer/data/observation/heII_paper/tol1924_416/img/hst_06708_01_wfpc2_f555w_pc_drz.fits',
        'file_name_r': '/home/benutzer/data/observation/heII_paper/tol1924_416/img/hst_06708_01_wfpc2_f814w_pc_drz.fits',
        'filter_name_b': 'F336W',
        'filter_name_g': 'F555W',
        'filter_name_r': 'F814W',
        'gamma_b': 2.8,
        'gamma_g': 2.8,
        'gamma_r': 2.8,
        'gamma_rgb': 2.8,
        'color_b': '#0096FF',
        'color_g': '#4CBB17',
        'color_r': '#FF4433',

        'file_name_align_g140l': '/home/benutzer/data/observation/heII_paper/tol1924_416/stis/O6ER18PEQ/o6er18peq_raw.fits',
        'file_name_stis_g140l': '/home/benutzer/data/observation/heII_paper/tol1924_416/stis/O6ER18020/o6er18020_flt.fits',
        'file_name_align_g430m': '/home/benutzer/data/observation/heII_paper/tol1924_416/stis/OE0708R7Q/oe0708r7q_raw.fits',
        'file_name_stis_g430m': '/home/benutzer/data/observation/heII_paper/tol1924_416/stis/OE0708020/oe0708020_flt.fits',
        'coords_img': '19h27m58.0s -41d34m33.0s',
        'coords_slit_center': '19h27m58.382s -41d34m30.94s',
        'coords_source_a': '19h27m58.382s -41d34m30.94s',
        #'coords_source_b': '19h27m57.90s -41d34m31.37s',
    }

}





rgb_image_he2_10, wcs_he2_10 = get_rgb_img(target='he2_10', cutout_size=(20, 20*1.5))
rgb_image_he2_10_zoom, wcs_he2_10_zoom = get_rgb_img(target='he2_10', cutout_size=(2, 2), coord_str='coords_source_b')
slit_dict_he2_10 = get_slit_pos(target='he2_10')

rgb_image_mrk33, wcs_mrk33 = get_rgb_img(target='mrk33', cutout_size=(20, 20*1.5))
rgb_image_mrk33_zoom, wcs_mrk33_zoom = get_rgb_img(target='mrk33', cutout_size=(6, 6), coord_str='coords_source_b')
slit_dict_mrk33 = get_slit_pos(target='mrk33')

rgb_image_ngc_3049, wcs_ngc_3049 = get_rgb_img(target='ngc_3049', cutout_size=(20, 20*1.5))
# rgb_image_ngc_3049_zoom, wcs_ngc_3049_zoom = get_single_rgb_img(target='ngc_3049', cutout_size=(6, 6), coord_str='coords_source_b')
rgb_image_ngc_3049_zoom, wcs_ngc_3049_zoom = get_rgb_img(target='ngc_3049', cutout_size=(6, 6), coord_str='coords_source_b')
slit_dict_ngc_3049 = get_slit_pos(target='ngc_3049')

rgb_image_ngc_3125, wcs_ngc_3125 = get_rgb_img(target='ngc_3125', cutout_size=(15, 15*1.5))
rgb_image_ngc_3125_zoom, wcs_ngc_3125_zoom = get_rgb_img(target='ngc_3125', cutout_size=(2, 2), coord_str='coords_source_a')
slit_dict_ngc_3125 = get_slit_pos(target='ngc_3125')

rgb_image_ngc_4214, wcs_ngc_4214 = get_rgb_img(target='ngc_4214', cutout_size=(25, 25*1.5))
rgb_image_ngc_4214_zoom, wcs_ngc_4214_zoom = get_rgb_img(target='ngc_4214', cutout_size=(3, 3), coord_str='coords_source_a')
slit_dict_ngc_4214 = get_slit_pos(target='ngc_4214')

rgb_image_ngc_4670, wcs_ngc_4670 = get_rgb_img(target='ngc_4670', cutout_size=(18, 18*1.5))
rgb_image_ngc_4670_zoom, wcs_ngc_4670_zoom = get_rgb_img(target='ngc_4670', cutout_size=(3, 3), coord_str='coords_source_a')
rgb_image_ngc_4670_zoom_2, wcs_ngc_4670_zoom_2 = get_rgb_img(target='ngc_4670', cutout_size=(4, 4), coord_str='coords_source_b')
slit_dict_ngc_4670 = get_slit_pos(target='ngc_4670')

rgb_image_tol89, wcs_tol89 = get_rgb_img(target='tol89', cutout_size=(25, 25*1.5))
rgb_image_tol89_zoom, wcs_tol89_zoom = get_rgb_img(target='tol89', cutout_size=(3, 3), coord_str='coords_source_a')
slit_dict_tol89 = get_slit_pos(target='tol89')

rgb_image_tol1924_416, wcs_tol1924_416 = get_rgb_img(target='tol1924_416', cutout_size=(13, 13*1.5))
rgb_image_tol1924_416_zoom, wcs_tol1924_416_zoom = get_rgb_img(target='tol1924_416', cutout_size=(3, 3), coord_str='coords_source_a')
slit_dict_tol1924_416 = get_slit_pos(target='tol1924_416')


coord_str='coords_source_b'

figure = plt.figure(figsize=(30, 40))
fontsize = 35
ax_img_he2_10 = figure.add_axes([0.01, 0.76, 0.52, 0.23], projection=wcs_he2_10)
ax_img_he2_10_zoom = figure.add_axes([0.33, 0.79, 0.1, 0.1], projection=wcs_he2_10_zoom)
ax_img_mrk33 = figure.add_axes([0.50, 0.76, 0.52, 0.23], projection=wcs_mrk33)
ax_img_mrk33_zoom = figure.add_axes([0.80, 0.82, 0.1, 0.1], projection=wcs_mrk33_zoom)
ax_img_ngc_3049 = figure.add_axes([0.01, 0.515, 0.52, 0.23], projection=wcs_ngc_3049)
ax_img_ngc_3049_zoom = figure.add_axes([0.11, 0.52, 0.1, 0.1], projection=wcs_ngc_3049_zoom)
ax_img_ngc_3125 = figure.add_axes([0.50, 0.515, 0.52, 0.23], projection=wcs_ngc_3125)
ax_img_ngc_3125_zoom = figure.add_axes([0.80, 0.52, 0.1, 0.1], projection=wcs_ngc_3125_zoom)
ax_img_ngc_4214 = figure.add_axes([0.01, 0.27, 0.52, 0.23], projection=wcs_ngc_4214)
ax_img_ngc_4214_zoom = figure.add_axes([0.35, 0.27, 0.1, 0.1], projection=wcs_ngc_4214_zoom)
ax_img_ngc_4670 = figure.add_axes([0.50, 0.27, 0.52, 0.23], projection=wcs_ngc_4670)
ax_img_ngc_4670_zoom = figure.add_axes([0.87, 0.38, 0.1, 0.1], projection=wcs_ngc_4670_zoom)
ax_img_ngc_4670_zoom_2 = figure.add_axes([0.57, 0.27, 0.1, 0.1], projection=wcs_ngc_4670_zoom_2)
ax_img_tol89 = figure.add_axes([0.01, 0.025, 0.52, 0.23], projection=wcs_tol89)
ax_img_tol89_zoom = figure.add_axes([0.35, 0.03, 0.1, 0.1], projection=wcs_tol89_zoom)
ax_img_tol1924_416 = figure.add_axes([0.50, 0.025, 0.52, 0.23], projection=wcs_tol1924_416)
ax_img_tol1924_416_zoom = figure.add_axes([0.75, 0.04, 0.1, 0.1], projection=wcs_tol1924_416_zoom)


ax_img_he2_10.imshow(rgb_image_he2_10)
ax_img_he2_10_zoom.imshow(rgb_image_he2_10_zoom)
display_slit(ax=ax_img_he2_10, target='he2_10', wcs=wcs_he2_10, data=rgb_image_he2_10[:,:,0],
             slit_dict=slit_dict_he2_10)
display_slit(ax=ax_img_he2_10_zoom, target='he2_10', wcs=wcs_he2_10_zoom, data=rgb_image_he2_10_zoom[:,:,0],
             slit_dict=slit_dict_he2_10)
display_circ(ax=ax_img_he2_10_zoom, wcs=wcs_he2_10_zoom, coords=file_name_dict['he2_10']['coords_source_a'],
             text='A', x_offset=-5, y_offset=7, fontsize=25)
display_circ(ax=ax_img_he2_10_zoom, wcs=wcs_he2_10_zoom, coords=file_name_dict['he2_10']['coords_source_b'],
             text='B', x_offset=-5, y_offset=7, fontsize=25)
display_circ(ax=ax_img_he2_10_zoom, wcs=wcs_he2_10_zoom, coords=file_name_dict['he2_10']['coords_source_c'],
             text='C', x_offset=-5, y_offset=7, fontsize=25)
display_circ(ax=ax_img_he2_10_zoom, wcs=wcs_he2_10_zoom, coords=file_name_dict['he2_10']['coords_source_d'],
             text='D', x_offset=-5, y_offset=7, fontsize=25)
display_bands(ax=ax_img_he2_10, target='he2_10', target_name='He 2-10', data=rgb_image_he2_10[:,:,0], fontsize=fontsize)
plotting_tools.arr_axis_params(ax=ax_img_he2_10, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label=' ', dec_axis_label='DEC. (2000.0)',
                               ra_minpad=0.3, dec_minpad=0.8, tick_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=4)
plotting_tools.arr_axis_params(ax=ax_img_he2_10_zoom, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label=' ', dec_axis_label=' ',
                               ra_minpad=0.3, dec_minpad=0.8, tick_color='white', label_color='white',
                               fontsize=fontsize-10, labelsize=fontsize-10, ra_tick_num=3, dec_tick_num=3)



ax_img_mrk33.imshow(rgb_image_mrk33)
ax_img_mrk33_zoom.imshow(rgb_image_mrk33_zoom)
display_slit(ax=ax_img_mrk33, target='mrk33', wcs=wcs_mrk33, data=rgb_image_mrk33[:,:,0],
             slit_dict=slit_dict_mrk33)
display_slit(ax=ax_img_mrk33_zoom, target='mrk33', wcs=wcs_mrk33_zoom, data=rgb_image_mrk33_zoom[:,:,0],
             slit_dict=slit_dict_mrk33)
display_circ(ax=ax_img_mrk33_zoom, wcs=wcs_mrk33_zoom, coords=file_name_dict['mrk33']['coords_source_a'],
             text='A', x_offset=-5, y_offset=7, fontsize=25)
display_circ(ax=ax_img_mrk33_zoom, wcs=wcs_mrk33_zoom, coords=file_name_dict['mrk33']['coords_source_b'],
             text='B', x_offset=-5, y_offset=7, fontsize=25)
display_bands(ax=ax_img_mrk33, target='mrk33', target_name='MRK 33', data=rgb_image_mrk33[:,:,0], fontsize=fontsize)
plotting_tools.arr_axis_params(ax=ax_img_mrk33, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label=' ', dec_axis_label=' ',
                               ra_minpad=0.3, dec_minpad=0.8, tick_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_img_mrk33_zoom, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label=' ', dec_axis_label=' ',
                               ra_minpad=0.3, dec_minpad=0.8, tick_color='white', label_color='white',
                               fontsize=fontsize - 14, labelsize=fontsize - 14, ra_tick_num=3, dec_tick_num=3)




ax_img_ngc_3049.imshow(rgb_image_ngc_3049)
ax_img_ngc_3049_zoom.imshow(rgb_image_ngc_3049_zoom)
display_slit(ax=ax_img_ngc_3049, target='ngc_3049', wcs=wcs_ngc_3049, data=rgb_image_ngc_3049[:,:,0],
             slit_dict=slit_dict_ngc_3049)
display_slit(ax=ax_img_ngc_3049_zoom, target='ngc_3049', wcs=wcs_ngc_3049_zoom, data=rgb_image_ngc_3049_zoom[:,:,0],
             slit_dict=slit_dict_ngc_3049)
display_circ(ax=ax_img_ngc_3049_zoom, wcs=wcs_ngc_3049_zoom, coords=file_name_dict['ngc_3049']['coords_source_a'],
             text='A', x_offset=-5, y_offset=7, fontsize=25)
display_circ(ax=ax_img_ngc_3049_zoom, wcs=wcs_ngc_3049_zoom, coords=file_name_dict['ngc_3049']['coords_source_b'],
             text='B', x_offset=-5, y_offset=7, fontsize=25)
display_bands(ax=ax_img_ngc_3049, target='ngc_3049', target_name='NGC 3049', data=rgb_image_ngc_3049[:,:,0], fontsize=fontsize)
plotting_tools.arr_axis_params(ax=ax_img_ngc_3049, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label=' ', dec_axis_label='DEC. (2000.0)',
                               ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_img_ngc_3049_zoom, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label=' ', dec_axis_label=' ',
                               ra_minpad=0.3, dec_minpad=0.8, tick_color='white', label_color='white',
                               fontsize=fontsize - 14, labelsize=fontsize - 14, ra_tick_num=3, dec_tick_num=3)



ax_img_ngc_3125.imshow(rgb_image_ngc_3125)
ax_img_ngc_3125_zoom.imshow(rgb_image_ngc_3125_zoom)
display_slit(ax=ax_img_ngc_3125, target='ngc_3125', wcs=wcs_ngc_3125, data=rgb_image_ngc_3125[:,:,0],
             slit_dict=slit_dict_ngc_3125)
display_slit(ax=ax_img_ngc_3125_zoom, target='ngc_3125', wcs=wcs_ngc_3125_zoom, data=rgb_image_ngc_3125_zoom[:,:,0],
             slit_dict=slit_dict_ngc_3125)
display_circ(ax=ax_img_ngc_3125_zoom, wcs=wcs_ngc_3125_zoom, coords=file_name_dict['ngc_3125']['coords_source_a'],
             text='A', x_offset=-5, y_offset=7, fontsize=25)
# display_circ(ax=ax_img_ngc_3125_zoom, wcs=wcs_ngc_3125_zoom, coords=file_name_dict['ngc_3125']['coords_source_b'],
#              text='B', x_offset=-5, y_offset=7, fontsize=25)
display_bands(ax=ax_img_ngc_3125, target='ngc_3125', target_name='NGC 3125', data=rgb_image_ngc_3125[:,:,0], fontsize=fontsize)
plotting_tools.arr_axis_params(ax=ax_img_ngc_3125, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label=' ', dec_axis_label=' ',
                               ra_minpad=0.3, dec_minpad=0.8, tick_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_img_ngc_3125_zoom, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label=' ', dec_axis_label=' ',
                               ra_minpad=0.3, dec_minpad=0.8, tick_color='white', label_color='white',
                               fontsize=fontsize - 14, labelsize=fontsize - 14, ra_tick_num=3, dec_tick_num=3)



ax_img_ngc_4214.imshow(rgb_image_ngc_4214)
ax_img_ngc_4214_zoom.imshow(rgb_image_ngc_4214_zoom)
display_slit(ax=ax_img_ngc_4214, target='ngc_4214', wcs=wcs_ngc_4214, data=rgb_image_ngc_4214[:,:,0],
             slit_dict=slit_dict_ngc_4214)
display_slit(ax=ax_img_ngc_4214_zoom, target='ngc_4214', wcs=wcs_ngc_4214_zoom, data=rgb_image_ngc_4214_zoom[:,:,0],
             slit_dict=slit_dict_ngc_4214)
display_circ(ax=ax_img_ngc_4214_zoom, wcs=wcs_ngc_4214_zoom, coords=file_name_dict['ngc_4214']['coords_source_a'],
             text='A', x_offset=-5, y_offset=7, fontsize=25)
# display_circ(ax=ax_img_ngc_4214_zoom, wcs=wcs_ngc_4214_zoom, coords=file_name_dict['ngc_4214']['coords_source_b'],
#              text='B', x_offset=-5, y_offset=7, fontsize=25)
display_bands(ax=ax_img_ngc_4214, target='ngc_4214', target_name='NGC 4214', data=rgb_image_ngc_4214[:,:,0], fontsize=fontsize)
plotting_tools.arr_axis_params(ax=ax_img_ngc_4214, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label=' ', dec_axis_label='DEC. (2000.0)',
                               ra_minpad=0.3, dec_minpad=0.5, tick_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_img_ngc_4214_zoom, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label=' ', dec_axis_label=' ',
                               ra_minpad=0.3, dec_minpad=0.8, tick_color='white', label_color='white',
                               fontsize=fontsize - 14, labelsize=fontsize - 14, ra_tick_num=3, dec_tick_num=3)





ax_img_ngc_4670.imshow(rgb_image_ngc_4670)
ax_img_ngc_4670_zoom.imshow(rgb_image_ngc_4670_zoom)
ax_img_ngc_4670_zoom_2.imshow(rgb_image_ngc_4670_zoom_2)
display_slit(ax=ax_img_ngc_4670, target='ngc_4670', wcs=wcs_ngc_4670, data=rgb_image_ngc_4670[:,:,0],
             slit_dict=slit_dict_ngc_4670)
display_slit(ax=ax_img_ngc_4670_zoom, target='ngc_4670', wcs=wcs_ngc_4670_zoom, data=rgb_image_ngc_4670_zoom[:,:,0],
             slit_dict=slit_dict_ngc_4670)
display_slit(ax=ax_img_ngc_4670_zoom_2, target='ngc_4670', wcs=wcs_ngc_4670_zoom_2, data=rgb_image_ngc_4670_zoom_2[:,:,0],
             slit_dict=slit_dict_ngc_4670)
display_circ(ax=ax_img_ngc_4670_zoom, wcs=wcs_ngc_4670_zoom, coords=file_name_dict['ngc_4670']['coords_source_a'],
             text='A', x_offset=-5, y_offset=7, fontsize=25)
display_circ(ax=ax_img_ngc_4670_zoom_2, wcs=wcs_ngc_4670_zoom_2, coords=file_name_dict['ngc_4670']['coords_source_b'],
             text='B', x_offset=-5, y_offset=7, fontsize=25)
display_circ(ax=ax_img_ngc_4670_zoom_2, wcs=wcs_ngc_4670_zoom_2, coords=file_name_dict['ngc_4670']['coords_source_c'],
             text='C', x_offset=-5, y_offset=7, fontsize=25)
display_bands(ax=ax_img_ngc_4670, target='ngc_4670', target_name='NGC 4670', data=rgb_image_ngc_4670[:,:,0], fontsize=fontsize)
plotting_tools.arr_axis_params(ax=ax_img_ngc_4670, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label=' ', dec_axis_label=' ',
                               ra_minpad=0.3, dec_minpad=0.8, tick_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_img_ngc_4670_zoom, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label=' ', dec_axis_label=' ',
                               ra_minpad=0.3, dec_minpad=0.8, tick_color='white', label_color='white',
                               fontsize=fontsize - 14, labelsize=fontsize - 14, ra_tick_num=3, dec_tick_num=3)

plotting_tools.arr_axis_params(ax=ax_img_ngc_4670_zoom_2, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label=' ', dec_axis_label=' ',
                               ra_minpad=0.3, dec_minpad=0.8, tick_color='white', label_color='white',
                               fontsize=fontsize - 14, labelsize=fontsize - 14, ra_tick_num=3, dec_tick_num=3)


ax_img_tol89.imshow(rgb_image_tol89)
ax_img_tol89_zoom.imshow(rgb_image_tol89_zoom)
display_slit(ax=ax_img_tol89, target='tol89', wcs=wcs_tol89, data=rgb_image_tol89[:,:,0],
             slit_dict=slit_dict_tol89)
display_slit(ax=ax_img_tol89_zoom, target='tol89', wcs=wcs_tol89_zoom, data=rgb_image_tol89_zoom[:,:,0],
             slit_dict=slit_dict_tol89)
display_circ(ax=ax_img_tol89_zoom, wcs=wcs_tol89_zoom, coords=file_name_dict['tol89']['coords_source_a'],
             text='A', x_offset=-5, y_offset=7, fontsize=25)
display_bands(ax=ax_img_tol89, target='tol89', target_name='TOL 89', data=rgb_image_tol89[:,:,0], fontsize=fontsize)
plotting_tools.arr_axis_params(ax=ax_img_tol89, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label='R.A. (2000.0)', dec_axis_label='DEC. (2000.0)',
                               ra_minpad=0.3, dec_minpad=0.8, tick_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_img_tol89_zoom, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label=' ', dec_axis_label=' ',
                               ra_minpad=0.3, dec_minpad=0.8, tick_color='white', label_color='white',
                               fontsize=fontsize - 14, labelsize=fontsize - 14, ra_tick_num=3, dec_tick_num=3)


ax_img_tol1924_416.imshow(rgb_image_tol1924_416)
ax_img_tol1924_416_zoom.imshow(rgb_image_tol1924_416_zoom)
display_slit(ax=ax_img_tol1924_416, target='tol1924_416', wcs=wcs_tol1924_416, data=rgb_image_tol1924_416[:,:,0],
             slit_dict=slit_dict_tol1924_416)
display_slit(ax=ax_img_tol1924_416_zoom, target='tol1924_416', wcs=wcs_tol1924_416_zoom, data=rgb_image_tol1924_416_zoom[:,:,0],
             slit_dict=slit_dict_tol1924_416)
display_circ(ax=ax_img_tol1924_416_zoom, wcs=wcs_tol1924_416_zoom, coords=file_name_dict['tol1924_416']['coords_source_a'],
             text='A', x_offset=-5, y_offset=7, fontsize=25)
display_bands(ax=ax_img_tol1924_416, target='tol1924_416', target_name='TOL 1924-416', data=rgb_image_tol1924_416[:,:,0], fontsize=fontsize)
plotting_tools.arr_axis_params(ax=ax_img_tol1924_416, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label='R.A. (2000.0)', dec_axis_label=' ',
                               ra_minpad=0.3, dec_minpad=0.8, tick_color='white',
                               fontsize=fontsize, labelsize=fontsize, ra_tick_num=5, dec_tick_num=3)
plotting_tools.arr_axis_params(ax=ax_img_tol1924_416_zoom, ra_tick_label=True, dec_tick_label=True,
                               ra_axis_label=' ', dec_axis_label=' ',
                               ra_minpad=0.3, dec_minpad=0.8, tick_color='white', label_color='white',
                               fontsize=fontsize - 14, labelsize=fontsize - 14, ra_tick_num=3, dec_tick_num=3)


plt.savefig('plot_output/slit_pos.png')
plt.savefig('plot_output/slit_pos.pdf')
