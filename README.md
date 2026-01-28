# KMORFS - Kinetic Modeling Of Residual Film Stress

Physics-informed machine learning framework for modeling residual stress evolution in thin film materials during Physical Vapor Deposition (PVD).

## Overview

KMORFS uses a three-term physics-based stress equation combined with PyTorch neural network optimization to predict stress-thickness relationships in thin films. The model captures:

1. **Kinetic stress** - Grain boundary effects during film growth
2. **Grain growth stress** - Atomic diffusion driven microstructure evolution
3. **Energetic stress** - Surface/interface energy contributions

## Installation

This repository is self-contained. No external KMORFS package required.

```bash
# Clone the repository
git clone https://github.com/[your-username]/KMORFS.git
cd KMORFS

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
python fit_alloy_stress.py
```

This will:
1. Load experimental data from `data/`
2. Fit the physics model using LBFGS optimization
3. Save optimized parameters to `output/optimized_parameters.csv`
4. Generate plots in `output/fitting_result.jpg`

### Custom Data

1. Add your experimental data as tab-separated text files in `data/` with columns:
   - `thickness` - Film thickness (nm)
   - `StressThickness` - Stress × thickness product (GPa·nm)
   - `R` - Deposition rate (nm/s)
   - `T` - Temperature (K)
   - `P` - Pressure (Pa)
   - `Melting_T` - Material melting temperature (K)

2. Edit `config.csv` to specify:
   - Dataset filenames
   - Process conditions (R, T, P, Melting_T)
   - Initial parameter guesses for each material

3. Run `python fit_alloy_stress.py`

## Project Structure

```
KMORFS/
├── fit_alloy_stress.py    # Main fitting script
├── config.csv             # Dataset and parameter configuration
├── kmorfs/                # Core physics module (self-contained)
│   ├── __init__.py
│   ├── stress_equation.py # Three-term stress model
│   ├── model.py           # PyTorch STFModelTorch
│   ├── alloy_extension.py # Rule of mixtures for alloys
│   └── data_utils.py      # Data loading utilities
├── data/                  # Experimental data files
└── output/                # Results (parameters, plots)
```

## Configuration

### config.csv Format

| Column | Description |
|--------|-------------|
| Fit_data | Filename of experimental data |
| R | Deposition rate (nm/s) |
| T | Temperature (K) |
| P | Pressure (Pa) |
| Melting_T | Melting temperature (K) - identifies material |
| K0 | Initial stress offset (MPa) |
| alpha1, L0, GrainSize_200 | Grain growth parameters |
| Sigma0, BetaD, Mfda, Di | Kinetic stress parameters |
| A0, B0, l0 | Energetic stress parameters |

**Note:** The first 4 unique materials in config.csv must be pure elements (e.g., Cr, V, Mo, W). Alloy properties (A0, B0, l0) are computed automatically via rule of mixtures.

### Supported Materials

- **Pure metals**: Cr, V, Mo, W, Cu, Ni, Ti, Co, Fe, Ag
- **Binary alloys**: CrW, WV, MoV and variants (e.g., Cr3W, WV3)

## Physics Model

The instantaneous stress σ(h) at thickness h is:

```
σ = σ_kinetic + σ_grain_growth + σ_energetic
```

Where:
- **Kinetic**: `σ_C + (σ_0/√L - σ_C) × exp(-β_D/(L×R))`
- **Grain growth**: `M_f×D_a × (L_bottom - L_0) / (L_bottom × L_surface)`
- **Energetic**: Pressure-dependent surface energy contribution

The stress-thickness product is obtained by integration:
```
S(h) = K_0 + ∫₀ʰ σ(h') dh'
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```
[Citation to be added upon publication]
```

## Acknowledgments

Developed at the Chason Lab, Brown University.
