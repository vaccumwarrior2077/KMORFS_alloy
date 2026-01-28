"""
Physics-based stress equation for thin film growth.

The model computes instantaneous stress as the sum of three components:
1. Kinetic stress - grain boundary effects during deposition
2. Grain growth stress - atomic diffusion driven microstructure evolution
3. Energetic stress - surface/interface energy contributions
"""

import numpy as np
import torch


def stress_equation(params, R, P, T, film_thickness):
    """
    Compute instantaneous film stress using the three-term physics model.

    Parameters
    ----------
    params : dict
        Model parameters:
        - SigmaC: Compressive stress limit (GPa)
        - Sigma0: Tensile stress coefficient (GPa)
        - BetaD: Diffusion coefficient
        - Ea: Activation energy (eV)
        - L0: Initial grain size (nm)
        - GrainSize_200: Grain size at 200nm thickness (nm)
        - alpha1: Grain growth rate coefficient
        - Mfda: Grain growth stress coefficient
        - Di: Energetic diffusion length (nm)
        - A0, B0, l0: Energetic stress parameters
    R : float or tensor
        Deposition rate (nm/s)
    P : float or tensor
        Pressure (Pa)
    T : float or tensor
        Temperature (K)
    film_thickness : float or tensor
        Current film thickness (nm)

    Returns
    -------
    tuple
        (total_stress, kinetic, grain_growth, energetic) - all in GPa
    """
    kB = 8.6173324e-5  # Boltzmann constant in eV/K
    GrainSize_Ref = 1.0  # Reference grain size (nm)

    # Extract parameters with physical constraints
    sigC = min(params['SigmaC'], 0)  # Must be compressive (negative)
    sig0 = max(params['Sigma0'], 0)  # Must be positive
    BetaD0 = max(params['BetaD'], 0)
    ea = max(params['Ea'], 0)
    GrainS_0 = params['L0']
    MfDa = params['Mfda']
    GrainSize_200 = params['GrainSize_200']
    alpha1 = params['alpha1']

    # Temperature-dependent diffusion
    if torch.is_tensor(T):
        BetaD_T = BetaD0 / (kB * T) * torch.exp(-ea / (kB * T))
    else:
        BetaD_T = BetaD0 / (kB * T) * np.exp(-ea / (kB * T))

    # Grain size evolution
    if GrainSize_200 > GrainS_0:
        alpha2 = 2 * (GrainSize_200 - GrainS_0) / 200.0 - alpha1
        if torch.is_tensor(T):
            alpha2 = torch.where(alpha2 < alpha1, (GrainSize_200 - GrainS_0) / 200.0, alpha2)
            alpha1 = torch.where(alpha2 < alpha1, alpha2, alpha1)
        else:
            alpha2 = np.where(alpha2 < alpha1, (GrainSize_200 - GrainS_0) / 200.0, alpha2)
            alpha1 = np.where(alpha2 < alpha1, alpha2, alpha1)
        GrainS_surface = GrainS_0 + alpha2 * film_thickness
    else:
        GrainS_surface = GrainSize_200

    GrainS_bottom = GrainS_0 + alpha1 * film_thickness

    # Kinetic stress component
    if torch.is_tensor(T):
        GrainS_bottom = torch.minimum(GrainS_bottom, GrainS_surface)
        Kinetic = sigC + (sig0 * torch.sqrt(GrainSize_Ref / GrainS_surface) - sigC) * \
                  torch.exp(-BetaD_T / (GrainS_surface * R))
    else:
        GrainS_bottom = np.minimum(GrainS_bottom, GrainS_surface)
        Kinetic = sigC + (sig0 * np.sqrt(GrainSize_Ref / GrainS_surface) - sigC) * \
                  np.exp(-BetaD_T / (GrainS_surface * R))

    # Grain growth stress component
    GrainGrowth = MfDa * (GrainS_bottom - GrainS_0) / (GrainS_bottom * GrainS_surface)

    # Energetic stress component (pressure-dependent)
    Energetic = 0.0
    if P != 0:
        Di = max(params['Di'], 0)
        A0 = min(params['A0'], 0)
        B0 = min(params['B0'], 0)
        l0 = max(params['l0'], 0)
        P0 = 1.0  # Reference pressure

        A_used = (1 - P / P0) * A0
        B_used = (1 - P / P0) * B0
        l_used = (1 - P / P0) * l0

        ratio = l_used / GrainS_surface

        if torch.is_tensor(T):
            part1 = torch.where(ratio > 1, A_used, ratio * A_used)
            part2 = torch.where(ratio > 1, 0.0,
                                (1 - ratio) * B_used * (1 - torch.exp(-R * l_used / Di)))
        else:
            part1 = np.where(ratio > 1, A_used, ratio * A_used)
            part2 = np.where(ratio > 1, 0.0,
                             (1 - ratio) * B_used * (1 - np.exp(-R * l_used / Di)))

        Energetic = part1 + part2

    return Kinetic + GrainGrowth + Energetic, Kinetic, GrainGrowth, Energetic
