import numpy as np
from dust_extinction.parameter_averages import CCM89
import astropy.units as u


class ExtinctionTools():

    def __int__(self):
        pass

    @staticmethod
    def c00_redd_curve(wavelength=6565, r_v=3.1):
        r"""
        calculate reddening curve
         following  Calzetti et al. (2000) doi:10.1086/308692
         using eq. 4

        :param wavelength: rest frame wavelength in angstrom of spectral part of which to compute the reddening curve
        :type wavelength: float or int
        :param r_v: default 3.1  total extinction at V
        :type r_v: float

        :return extinction E(B - V) in mag
        :rtype: array_like
        """

        # change wavelength from Angstrom to microns
        wavelength *= 1e-4

        # eq. 4
        if (wavelength > 0.63) & (wavelength < 2.20):
            # suitable for 0.63 micron < wavelength < 2.20 micron
            k_lambda = 2.659 * (-1.857 + 1.040/wavelength) + r_v
        elif (wavelength > 0.12) & (wavelength < 0.63):
            # suitable for 0.12 micron < wavelength < 0.63 micron
            k_lambda = 2.659 * (- 2.156 + 1.509 / wavelength - 0.198 / wavelength**2 + 0.011 / wavelength**3) + r_v
        else:
            raise KeyError('wavelength must be > 1200 Angstrom and < 22000 Angstrom')

        return k_lambda

    @staticmethod
    def calc_stellar_extinct(wavelength, ebv, r_v):
        return ExtinctionTools.compute_reddening_curve(wavelength=wavelength, r_v=r_v) * ebv

    @staticmethod
    def color_ext_ccm89_ebv(wave1, wave2, ebv, r_v=3.1):

        model_ccm89 = CCM89(Rv=r_v)
        reddening1 = model_ccm89(wave1*u.micron) * r_v
        reddening2 = model_ccm89(wave2*u.micron) * r_v

        return (reddening1 - reddening2)*ebv

    @staticmethod
    def ebv2av(ebv, r_v=3.1):
        wave_v = 5388.55 * 1e-4
        model_ccm89 = CCM89(Rv=r_v)
        return model_ccm89(wave_v*u.micron) * r_v * ebv

    @staticmethod
    def av2ebv(av, r_v=3.1):
        wave_v = 5388.55 * 1e-4
        model_ccm89 = CCM89(Rv=r_v)
        return av / (model_ccm89(wave_v*u.micron) * r_v)

    @staticmethod
    def color_ext_ccm89_av(wave1, wave2, av, r_v=3.1):

        model_ccm89 = CCM89(Rv=r_v)
        reddening1 = model_ccm89(wave1*u.micron) * r_v
        reddening2 = model_ccm89(wave2*u.micron) * r_v

        wave_v = 5388.55 * 1e-4
        reddening_v = model_ccm89(wave_v*u.micron) * r_v

        return (reddening1 - reddening2)*av/reddening_v

    @staticmethod
    def get_balmer_extinct_alpha_beta(flux_h_alpha_6565, flux_h_beta_4863):
        r"""
        calculate gas extinction using the Balmer decrement
         following dominguez+13 doi:10.1088/0004-637X/763/2/145
         assuming an intrinsic ratio H\alpha/H\beta=2.85 (Osterbrock 1989)
         and the Whitford reddening curve from Miller & Mathews (1972)

        :return extinction E(B - V) in mag
        :rtype: array_like
        """

        # dominguez et al 2013 doi:10.1088/0004-637X/763/2/145
        # eq. 4
        # e_b_v = 1.97 * np.log10(flux_h_alpha_6565 / flux_h_beta_4863) - 1.97 * np.log10(2.86)
        # e_b_v = 1.97 * np.log10((flux_h_alpha_6565 / flux_h_beta_4863) / 2.86 )
        e_b_v = 1.97 * np.log10((flux_h_alpha_6565 / flux_h_beta_4863) / 2.83)

        return e_b_v


    @staticmethod
    def get_balmer_extinct_alpha_beta_err(flux_h_alpha_6565, flux_h_beta_4863, flux_h_alpha_6565_err, flux_h_beta_4863_err):
        r"""
        calculate gas extinction using the Balmer decrement
         following dominguez+13 doi:10.1088/0004-637X/763/2/145
         assuming an intrinsic ratio H\alpha/H\beta=2.85 (Osterbrock 1989)
         and the Whitford reddening curve from Miller & Mathews (1972)

        :return extinction E(B - V) in mag
        :rtype: array_like
        """

        # dominguez et al 2013 doi:10.1088/0004-637X/763/2/145
        # eq. 4
        # simple err prop
        ebv_err = np.sqrt((1.97 * 2.83 * flux_h_alpha_6565_err / (flux_h_alpha_6565 * np.log(10)))**2 +
                          (1.97 * 2.83 * flux_h_beta_4863_err / (flux_h_beta_4863 * np.log(10)))**2)

        return ebv_err

    @staticmethod
    def get_balmer_extinct_beta_gamma(flux_h_beta_4863, flux_h_gamma_4342):
        r"""
        See appendix in Momcheva et al. 2013 doi:10.1088/0004-6256/145/2/47

        :return extinction E(B - V) in mag
        :rtype: array_like
        """

        r_v = 3.1

        model_ccm89 = CCM89(Rv=r_v)
        kappa_h_beta = model_ccm89(4862.692*1e-4*u.micron) * r_v
        kappa_h_gamma = model_ccm89(4341.692*1e-4*u.micron) * r_v
        h_gamma_over_h_beta = 0.469

        e_b_v = (-2.5 / (kappa_h_beta - kappa_h_gamma)) * np.log10(h_gamma_over_h_beta / (flux_h_gamma_4342 / flux_h_beta_4863))

        return e_b_v

    @staticmethod
    def get_balmer_extinct_beta_gamma_err(flux_h_beta_4863, flux_h_gamma_4342, flux_h_beta_4863_err, flux_h_gamma_4342_err):
        r"""
        See appendix in Momcheva et al. 2013 doi:10.1088/0004-6256/145/2/47

        :return extinction E(B - V) in mag
        :rtype: array_like
        """

        r_v = 3.1

        model_ccm89 = CCM89(Rv=r_v)
        kappa_h_beta = model_ccm89(4862.692*1e-4*u.micron) * r_v
        kappa_h_gamma = model_ccm89(4341.692*1e-4*u.micron) * r_v
        h_gamma_over_h_beta = 0.469

        ebv_err = np.sqrt(((-2.5 / (kappa_h_beta - kappa_h_gamma)) * h_gamma_over_h_beta * flux_h_beta_4863_err / (flux_h_beta_4863 * np.log(10)))**2 +
                          ((-2.5 / (kappa_h_beta - kappa_h_gamma)) * h_gamma_over_h_beta * flux_h_gamma_4342_err / (flux_h_gamma_4342 * np.log(10)))**2)

        return ebv_err

    def get_corr_h_alpha_flux(self, flux_h_alpha_6565=None, flux_h_beta_4863=None, line_shape='gauss'):
        """
        Get extinction corrected h_alpha flux err following Calzetti et al. (2000) doi:10.1086/308692
         using eq. 2 and eq.3
        """

        # get eimission line flux
        if (flux_h_alpha_6565 is None) & (flux_h_beta_4863 is None):
            flux_h_alpha_6565 = self.get_emission_line_flux(line_wavelength=6565, line_shape=line_shape)
            flux_h_beta_4863 = self.get_emission_line_flux(line_wavelength=4863, line_shape=line_shape)

        # get extinction
        e_b_v = self.get_extinction(flux_h_alpha_6565=flux_h_alpha_6565, flux_h_beta_4863=flux_h_beta_4863,
                                    line_shape=line_shape)

        #  the color excess of the stellar continuum is linked to the color excess e_s_b_v derived from the nebular
        #  gas emission lines e_b_v
        # correcting using eq. 3
        e_s_b_v = 0.44 * e_b_v

        # get reddening curve
        k_h_alpha = self.compute_reddening_curve(wavelength=6565, r_v=3.1)

        # flux crrection eq. 2
        corr_flux_h_alpha_6565 = flux_h_alpha_6565 * 10 ** (0.4 * e_s_b_v * k_h_alpha)

        return corr_flux_h_alpha_6565

    def get_corr_h_alpha_flux_err(self, flux_h_alpha_6565=None, flux_h_beta_4863=None, flux_h_alpha_6565_err=None,
                                       flux_h_beta_4863_err=None, line_shape='gauss'):
        """
        Get extinction corrected h_alpha flux following Calzetti et al. (2000) doi:10.1086/308692
         using eq. 2 and eq.3
        """

        # get eimission line flux
        if (flux_h_alpha_6565 is None) & (flux_h_beta_4863 is None):
            flux_h_alpha_6565 = self.get_emission_line_flux(line_wavelength=6565, line_shape=line_shape)
            flux_h_beta_4863 = self.get_emission_line_flux(line_wavelength=4863, line_shape=line_shape)

            flux_h_alpha_6565_err = self.get_emission_line_flux_err(line_wavelength=6565, line_shape=line_shape)
            flux_h_beta_4863_err = self.get_emission_line_flux_err(line_wavelength=4863, line_shape=line_shape)

        # get extinction
        e_b_v = self.get_extinction(flux_h_alpha_6565=flux_h_alpha_6565, flux_h_beta_4863=flux_h_beta_4863,
                                    line_shape=line_shape)
        # get extinction err
        e_b_v_err = self.get_extinction_err(flux_h_alpha_6565=flux_h_alpha_6565, flux_h_beta_4863=flux_h_beta_4863,
                                            flux_h_alpha_6565_err=flux_h_alpha_6565_err,
                                            flux_h_beta_4863_err=flux_h_beta_4863_err, line_shape=line_shape)

        # the color excess of the stellar continuum is linked to the color excess e_s_b_v derived from the nebular
        # gas emission lines e_b_v
        # correcting using eq. 3
        e_s_b_v = 0.44 * e_b_v
        # err
        e_s_b_v_err = 0.44 * e_b_v_err

        # get reddening curve
        k_h_alpha = self.compute_reddening_curve(wavelength=6565, r_v=3.1)

        # errorpropagation of eq. 2
        corr_flux_h_alpha_6565_err = np.sqrt((flux_h_alpha_6565_err * 10 ** (0.4 * e_s_b_v * k_h_alpha)) ** 2 +
                                             (e_s_b_v_err * flux_h_alpha_6565 * 10 ** (0.4 * e_s_b_v * k_h_alpha) *
                                              np.log(10) * 0.4 * k_h_alpha) ** 2)

        return corr_flux_h_alpha_6565_err

    def get_corr_h_alpha_lum(self, flux_h_alpha_6565=None, flux_h_beta_4863=None, redshift=None, line_shape='gauss'):
        """
         as described in Kewley et al. 2002 doi:10.1086/344487
        :param line_shape: line shape as described in get_emission_line_flux()
        :return: float or array
        """
        # get eimission line flux
        if (flux_h_alpha_6565 is None) & (flux_h_beta_4863 is None):
            flux_h_alpha_6565 = self.get_emission_line_flux(line_wavelength=6565, line_shape=line_shape)
            flux_h_beta_4863 = self.get_emission_line_flux(line_wavelength=4863, line_shape=line_shape)

        corrected_flux = np.array(self.get_corr_h_alpha_flux(flux_h_alpha_6565=flux_h_alpha_6565,
                                                             flux_h_beta_4863=flux_h_beta_4863, line_shape=line_shape),
                                  dtype=np.float64)

        if redshift is None:
            redshift = self.get_redshift()

        luminosity_dist = np.array(self.cosmology.luminosity_distance(redshift).to(u.cm).value, dtype=np.float64)

        corr_h_alpha_lum = corrected_flux * (1e-17 * 4 * np.pi) * (luminosity_dist**2)

        return corr_h_alpha_lum

    def get_corr_h_alpha_lum_err(self, flux_h_alpha_6565=None, flux_h_beta_4863=None, flux_h_alpha_6565_err=None,
                                 flux_h_beta_4863_err=None, redshift=None, line_shape='gauss'):
        """
         as described in Kewley et al. 2002 doi:10.1086/344487
        :param line_shape: line shape as described in get_emission_line_flux()
        :return: float or array
        """
        # get eimission line flux
        if (flux_h_alpha_6565 is None) & (flux_h_beta_4863 is None):
            flux_h_alpha_6565 = self.get_emission_line_flux(line_wavelength=6565, line_shape=line_shape)
            flux_h_beta_4863 = self.get_emission_line_flux(line_wavelength=4863, line_shape=line_shape)

            flux_h_alpha_6565_err = self.get_emission_line_flux_err(line_wavelength=6565, line_shape=line_shape)
            flux_h_beta_4863_err = self.get_emission_line_flux_err(line_wavelength=4863, line_shape=line_shape)

        corrected_flux_err = np.array(self.get_corr_h_alpha_flux_err(flux_h_alpha_6565=flux_h_alpha_6565,
                                                                     flux_h_beta_4863=flux_h_beta_4863,
                                                                     flux_h_alpha_6565_err=flux_h_alpha_6565_err,
                                                                     flux_h_beta_4863_err=flux_h_beta_4863_err,
                                                                     line_shape=line_shape), dtype=np.float64)
        if redshift is None:
            redshift = self.get_redshift()

        luminosity_dist = np.array(self.cosmology.luminosity_distance(redshift).to(u.cm).value, dtype=np.float64)

        corr_h_alpha_lum_err = corrected_flux_err * (1e-17 * 4 * np.pi) * luminosity_dist * luminosity_dist

        return corr_h_alpha_lum_err

    def get_log_corr_h_alpha_lum(self, flux_h_alpha_6565=None, flux_h_beta_4863=None, redshift=None,
                                 line_shape='gauss'):
        """
         as described in Kewley et al. 2002 doi:10.1086/344487
        :param line_shape: line shape as described in get_emission_line_flux()
        :return: float or array
        """
        # get eimission line flux
        if (flux_h_alpha_6565 is None) & (flux_h_beta_4863 is None):
            flux_h_alpha_6565 = self.get_emission_line_flux(line_wavelength=6565, line_shape=line_shape)
            flux_h_beta_4863 = self.get_emission_line_flux(line_wavelength=4863, line_shape=line_shape)

        log_corrected_flux = np.log10(self.get_corr_h_alpha_flux(flux_h_alpha_6565=flux_h_alpha_6565,
                                                                      flux_h_beta_4863=flux_h_beta_4863,
                                                                      line_shape=line_shape))

        if redshift is None:
            redshift = self.get_redshift()

        log_lum_dist = np.log10(self.cosmology.luminosity_distance(redshift).to(u.cm).value)

        log_corr_h_alpha_lum = log_corrected_flux + np.log10(1e-17 * 4 * np.pi) + 2 * log_lum_dist

        return log_corr_h_alpha_lum

    def get_log_corr_h_alpha_lum_err(self, flux_h_alpha_6565=None, flux_h_beta_4863=None, flux_h_alpha_6565_err=None,
                                     flux_h_beta_4863_err=None, line_shape='gauss'):
        """
         as described in Kewley et al. 2002 doi:10.1086/344487
        :param line_shape: line shape as described in get_emission_line_flux()
        :return: float or array
        """
        # get eimission line flux
        if (flux_h_alpha_6565 is None) & (flux_h_beta_4863 is None):
            flux_h_alpha_6565 = self.get_emission_line_flux(line_wavelength=6565, line_shape=line_shape)
            flux_h_beta_4863 = self.get_emission_line_flux(line_wavelength=4863, line_shape=line_shape)

            flux_h_alpha_6565_err = self.get_emission_line_flux_err(line_wavelength=6565, line_shape=line_shape)
            flux_h_beta_4863_err = self.get_emission_line_flux_err(line_wavelength=4863, line_shape=line_shape)

        corrected_flux = self.get_corr_h_alpha_flux(flux_h_alpha_6565=flux_h_alpha_6565,
                                                         flux_h_beta_4863=flux_h_beta_4863, line_shape=line_shape)

        corrected_flux_err = self.get_corr_h_alpha_flux_err(flux_h_alpha_6565=flux_h_alpha_6565,
                                                            flux_h_beta_4863=flux_h_beta_4863,
                                                            flux_h_alpha_6565_err=flux_h_alpha_6565_err,
                                                            flux_h_beta_4863_err=flux_h_beta_4863_err,
                                                            line_shape=line_shape)

        log_corr_h_alpha_lum_err = corrected_flux_err / (corrected_flux * np.log(10))

        return log_corr_h_alpha_lum_err

    def get_h_alpha_corr_sfr(self, flux_h_alpha_6565=None, flux_h_beta_4863=None, redshift=None,
                             line_shape='gauss', IMF='Salpeter'):
        """
        to calculate the SFR from the H\alpha emission line
         we need to correct the luminosity for  extinction-corrected
         as described in Kewley et al. 2002 doi:10.1086/344487
        :param line_shape: line shape as described in get_emission_line_flux()
        :return: float or array
        """
        if (flux_h_alpha_6565 is None) & (flux_h_beta_4863 is None):
            flux_h_alpha_6565 = self.get_emission_line_flux(line_wavelength=6565, line_shape=line_shape)
            flux_h_beta_4863 = self.get_emission_line_flux(line_wavelength=4863, line_shape=line_shape)

        if redshift is None:
            redshift=self.get_redshift()

        corr_h_alpha_lum = self.get_corr_h_alpha_lum(flux_h_alpha_6565=flux_h_alpha_6565,
                                                     flux_h_beta_4863=flux_h_beta_4863, redshift=redshift,
                                                     line_shape=line_shape)

        # Kewley et al. 2002 doi:10.1086/344487
        # equation 6
        # Salpeter IMF
        if IMF == 'Salpeter':
            sfr = 7.9 * 10 **(-42) * corr_h_alpha_lum
        elif IMF == 'Chabrier':
            sfr = 1.2 * 10 **(-41) * corr_h_alpha_lum

        return sfr

    def get_h_alpha_corr_sfr_err(self, flux_h_alpha_6565=None, flux_h_beta_4863=None,
                                     flux_h_alpha_6565_err=None, flux_h_beta_4863_err=None, line_shape='gauss'):
        """
        to calculate the SFR from the H\alpha emission line
         we need to correct the luminosity for  extinction-corrected
         as described in Kewley et al. 2002 doi:10.1086/344487
        :param line_shape: line shape as described in get_emission_line_flux()
        :return: float or array
        """
        # get emission line flux
        if (flux_h_alpha_6565 is None) & (flux_h_beta_4863 is None):
            flux_h_alpha_6565 = self.get_emission_line_flux(line_wavelength=6565, line_shape=line_shape)
            flux_h_beta_4863 = self.get_emission_line_flux(line_wavelength=4863, line_shape=line_shape)

            flux_h_alpha_6565_err = self.get_emission_line_flux_err(line_wavelength=6565, line_shape=line_shape)
            flux_h_beta_4863_err = self.get_emission_line_flux_err(line_wavelength=4863, line_shape=line_shape)

        corr_h_alpha_lum_err = self.get_corr_h_alpha_lum_err(flux_h_alpha_6565=flux_h_alpha_6565,
                                                             flux_h_beta_4863=flux_h_beta_4863,
                                                             flux_h_alpha_6565_err=flux_h_alpha_6565_err,
                                                             flux_h_beta_4863_err=flux_h_beta_4863_err,
                                                             line_shape=line_shape)

        # Kewley et al. 2002 doi:10.1086/344487
        # error propagation on equation 6
        sfr_err = 7.9 *1e-42 * corr_h_alpha_lum_err

        return sfr_err

    def get_log_h_alpha_corr_sfr(self, flux_h_alpha_6565=None, flux_h_beta_4863=None, redshift=None,
                                 line_shape='gauss', IMF='Salpeter'):
        """
        to calculate the SFR from the H\alpha emission line
         we need to correct the luminosity for  extinction-corrected
         as described in Kewley et al. 2002 doi:10.1086/344487
        :param line_shape: line shape as described in get_emission_line_flux()
        :return: float or array
        """
        if (flux_h_alpha_6565 is None) & (flux_h_beta_4863 is None):
            flux_h_alpha_6565 = self.get_emission_line_flux(line_wavelength=6565, line_shape=line_shape)
            flux_h_beta_4863 = self.get_emission_line_flux(line_wavelength=4863, line_shape=line_shape)

        if redshift is None:
            redshift=self.get_redshift()

        log_corr_h_alpha_lum = self.get_log_corr_h_alpha_lum(flux_h_alpha_6565=flux_h_alpha_6565,
                                                         flux_h_beta_4863=flux_h_beta_4863, redshift=redshift,
                                                             line_shape=line_shape)

        # Kewley et al. 2002 doi:10.1086/344487
        # equation 6
        # using a Salpeter IMF
        if IMF == 'Salpeter':
            log_sfr = np.log10(7.9) - 42 + log_corr_h_alpha_lum
        elif IMF == 'Chabrier':
            log_sfr = np.log10(1.2) - 41 + log_corr_h_alpha_lum
        elif IMF == 'Kroupa':
            log_sfr = np.log10(5.5) - 42 + log_corr_h_alpha_lum

        return log_sfr

    def get_log_h_alpha_corr_sfr_err(self, flux_h_alpha_6565=None, flux_h_beta_4863=None,
                                 flux_h_alpha_6565_err=None, flux_h_beta_4863_err=None, line_shape='gauss', IMF='Salpeter'):
        """
        to calculate the SFR from the H\alpha emission line
         we need to correct the luminosity for  extinction-corrected
         as described in Kewley et al. 2002 doi:10.1086/344487
        :param line_shape: line shape as described in get_emission_line_flux()
        :return: float or array
        """

        if (flux_h_alpha_6565 is None) & (flux_h_beta_4863 is None):
            flux_h_alpha_6565 = self.get_emission_line_flux(line_wavelength=6565, line_shape=line_shape)
            flux_h_beta_4863 = self.get_emission_line_flux(line_wavelength=4863, line_shape=line_shape)

            flux_h_alpha_6565_err = self.get_emission_line_flux_err(line_wavelength=6565, line_shape=line_shape)
            flux_h_beta_4863_err = self.get_emission_line_flux_err(line_wavelength=4863, line_shape=line_shape)

        log_corr_h_alpha_l_err = self.get_log_corr_h_alpha_lum_err(flux_h_alpha_6565=flux_h_alpha_6565,
                                                               flux_h_beta_4863=flux_h_beta_4863,
                                                               flux_h_alpha_6565_err=flux_h_alpha_6565_err,
                                                               flux_h_beta_4863_err=flux_h_beta_4863_err,
                                                               line_shape=line_shape)

        # Kewley et al. 2002 doi:10.1086/344487
        # equation 6 (using error propagation)
        # using a Salpeter IMF
        if IMF == 'Salpeter':
            log_sfr_err = np.log10(7.9) - 42 + log_corr_h_alpha_l_err
        elif IMF == 'Chabrier':
            log_sfr_err = np.log10(1.2) - 41 + log_corr_h_alpha_l_err

        return log_sfr_err
