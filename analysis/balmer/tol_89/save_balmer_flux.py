import numpy as np

# flux_h_beta = 10**(-12.86)
flux_h_beta = 10**(-13.32)

# flux_h_alpha = flux_h_beta * (390.50 / 100)
flux_h_alpha = flux_h_beta * (290.20 / 100)

balmer_dict = {
    'h_alpha_flux': flux_h_alpha,
    'h_beta_flux': flux_h_beta,
    'h_alpha_flux_err': flux_h_alpha*1e-2,
    'h_beta_flux_err': flux_h_beta*1e-2,
}

np.save('data_output/balmer_dict_TOL_89.npy', balmer_dict)



