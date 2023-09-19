from astropy.cosmology import LambdaCDM
import astropy.units as u
import numpy as np


cosmology = LambdaCDM(H0=70.0, Om0=0.3, Ode0=0.7)


mean_redshift = np.mean([2.574, 2.614, 3.202, 2.385, 3.077, 2.598])

lum_dist_0 = cosmology.luminosity_distance(0.0037).value
lum_dist = cosmology.luminosity_distance(mean_redshift).value

print('wave', 164 * (1 + mean_redshift))
print(60 * 60)
print('lum_dist_0 ', lum_dist_0)
print('lum_dist ', lum_dist)

flux_1640 = (23.76 * lum_dist_0 ** 2) / lum_dist ** 2
flux_4686 = (9.59 * lum_dist_0 ** 2) / lum_dist ** 2

print('flux_1640 ', flux_1640)
print('flux_4686 ', flux_4686)
