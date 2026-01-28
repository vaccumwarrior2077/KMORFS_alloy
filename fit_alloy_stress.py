"""
KMORFS - Kinetic Modeling Of Residual Film Stress

This script fits physics-based stress models to thin film deposition data
using PyTorch neural networks with LBFGS optimization.

Author: Tong Su
Affiliation: Brown University, Chason Lab
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

import torch
import torch.nn as nn

# Import from local kmorfs module (self-contained)
from kmorfs import RawData_extract, AlloyMaterialDependentExtension, STFModelTorch

# Configuration
matplotlib.rcParams['font.family'] = "Times New Roman"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths (relative to script location)
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"
CONFIG_FILE = SCRIPT_DIR / "config.csv"

# Plot colors
COLORS = np.array([
    "#5B9BD5",  # Soft Blue
    "#A5D6A7",  # Soft Green
    "#F1C40F",  # Warm Yellow
    "#E74C3C",  # Muted Red
    "#9B59B6",  # Soft Purple
    "#F39C12",  # Orange
    "#1F77B4",  # Cool Blue
    "#BDC3C7"   # Light Gray
])


def load_data(mainfile):
    """Load and preprocess experimental data."""
    path_info = (str(SCRIPT_DIR) + "/", str(CONFIG_FILE.name), "data/")
    dataset_name = mainfile["Fit_data"]
    fit_data, _ = RawData_extract(dataset_name, path_info, plot_setting=0)
    return fit_data


def setup_parameters(mainfile):
    """
    Setup initial parameters and bounds for optimization.

    Returns dict with all parameter arrays and settings.
    """
    # Process conditions from config
    process_condition = mainfile[['R', 'T', 'P', 'Melting_T']]

    # Initial guesses - only K0 for process, 10 params for materials
    initial_process = mainfile[['K0']]
    initial_materials = mainfile[['alpha1', 'L0', 'GrainSize_200', 'Sigma0',
                                   'BetaD', 'Mfda', 'Di', 'A0', 'B0', 'l0']].dropna()

    # Parameter names for output
    process_para_name = 'K0'
    materials_para_name = list(mainfile.columns[7:17])

    # Flatten to 1D vectors
    materials_1d = initial_materials.values.flatten()
    process_1d = initial_process.values.flatten()
    x_vector = np.concatenate([materials_1d, process_1d])

    # Bound multipliers for each material parameter
    # [alpha1, L0, GrainSize_200, Sigma0, BetaD, Mfda, Di, A0, B0, l0]
    materials_bound = np.array([3, 2, 0.5, 3, 2, 0.5, 2, 2, 0.5, 0.2])
    process_bound = np.array([300])  # K0 ± 300 MPa

    # Compute bounds for materials
    materials_lb = initial_materials.copy().astype(float)
    materials_ub = initial_materials.copy().astype(float)

    # Zero lower bound for: alpha1, L0, BetaD, Di
    for i in [0, 1, 4, 6]:
        mater_f1 = 0
        mater_f2 = initial_materials.iloc[:, i] * (1 + materials_bound[i])
        materials_lb.iloc[:, i] = np.minimum(mater_f1, mater_f2)
        materials_ub.iloc[:, i] = np.maximum(mater_f1, mater_f2)

    # Multiplicative bounds for: GrainSize_200, Sigma0, Mfda, A0, B0, l0
    for i in [2, 3, 5, 7, 8, 9]:
        mater_f1 = initial_materials.iloc[:, i] * (1 / (1 + materials_bound[i]))
        mater_f2 = initial_materials.iloc[:, i] * (1 + materials_bound[i])
        materials_lb.iloc[:, i] = np.minimum(mater_f1, mater_f2)
        materials_ub.iloc[:, i] = np.maximum(mater_f1, mater_f2)

    # Process bounds (K0 ± 300 MPa additive)
    process_lb = initial_process.copy().astype(float)
    process_ub = initial_process.copy().astype(float)
    process_lb.iloc[:, 0] = initial_process.iloc[:, 0] - process_bound[0]
    process_ub.iloc[:, 0] = initial_process.iloc[:, 0] + process_bound[0]

    # Flatten bounds
    para_lb = np.concatenate([materials_lb.values.flatten(), process_lb.values.flatten()])
    para_ub = np.concatenate([materials_ub.values.flatten(), process_ub.values.flatten()])

    # Scale initial vector to [0, 1]
    x_vector_scaled = (x_vector - para_lb) / (para_ub - para_lb)

    # file_setting: [n_pure_elements, n_process_params, n_material_params]
    # n_pure_elements controls alloy blending - set to 4 for Cr, V, Mo, W
    file_setting = [4, len(process_bound), len(materials_bound)]

    return {
        'x_vector_scaled': x_vector_scaled,
        'para_lb': para_lb,
        'para_ub': para_ub,
        'process_condition': process_condition,
        'initial_process': initial_process,
        'initial_materials': initial_materials,
        'process_para_name': process_para_name,
        'materials_para_name': materials_para_name,
        'materials_bound': materials_bound,
        'process_bound': process_bound,
        'file_setting': file_setting
    }


def train_model(model, x_tensor, y_tensor, process_tensor, fit_index_tensor,
                scaler_x, scaler_fity, epochs=26):
    """Train the model using LBFGS optimization."""
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=0.1,
        max_iter=20,
        history_size=10,
        line_search_fn="strong_wolfe"
    )
    loss_fn = nn.MSELoss()

    def closure():
        optimizer.zero_grad()
        y_pred = model(x_tensor, process_tensor, fit_index_tensor, scaler_x, scaler_fity)
        loss = loss_fn(y_pred, y_tensor)
        loss.backward()
        return loss

    print("Training model...")
    for epoch in range(epochs):
        loss = optimizer.step(closure)
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}, Loss: {loss.item() * 100:.5f}")

    return model


def plot_results(fit_data, process_condition, alloy_ext):
    """Generate and save result plots."""
    melting_vals = process_condition['Melting_T'].unique()
    n_plots = len(melting_vals)
    n_cols = min(4, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_plots == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, tm in enumerate(melting_vals):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]

        matching_rows = process_condition[process_condition['Melting_T'] == tm].index.tolist()

        color_id = 0
        for dataset_id in matching_rows:
            mask = fit_data["Index"] == dataset_id + 1
            thickness = fit_data.loc[mask, "thickness"].reset_index(drop=True)
            raw_data = fit_data.loc[mask, "StressThickness"].reset_index(drop=True)
            pred_data = fit_data.loc[mask, "y_pred"].reset_index(drop=True)

            ax.plot(thickness, pred_data, color=COLORS[color_id], linewidth=3)
            ax.scatter(thickness, raw_data, color=COLORS[color_id], s=10)
            color_id = (color_id + 1) % len(COLORS)

        ax.set_title(f"{alloy_ext.unique[i]} (Tm = {tm:.0f} K)")
        ax.set_xlabel("Thickness (nm)")
        ax.set_ylabel("Stress×Thickness (GPa·nm)")
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ax.tick_params(labelsize=12)

    # Hide unused axes
    for j in range(n_plots, n_rows * n_cols):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()

    OUTPUT_DIR.mkdir(exist_ok=True)
    filepath = OUTPUT_DIR / "fitting_result.jpg"
    fig.savefig(filepath, format="jpg", dpi=300)
    print(f"Saved plot to {filepath}")

    plt.show()
    return fig


def save_results(model, params, mainfile, alloy_ext, process_condition):
    """Extract optimized parameters and save to CSV."""
    with torch.no_grad():
        optimized_scaled = model.x_vector_scaled.cpu().numpy()

    para_lb, para_ub = params['para_lb'], params['para_ub']
    materials_bound = params['materials_bound']
    process_bound = params['process_bound']

    # Unscale parameters
    vector_param = optimized_scaled * (para_ub - para_lb) + para_lb

    # Get unique materials (by melting temperature)
    melting_temps = process_condition['Melting_T'].values
    unique_temps = list(dict.fromkeys(melting_temps))  # Preserve order
    num_materials = len(unique_temps)

    # Split into materials and process
    mat_count = num_materials * len(materials_bound)
    partial_materials = vector_param[:mat_count].reshape(num_materials, len(materials_bound))

    # Apply alloy extension
    materials_para = alloy_ext.alloy_extension(partial_materials, partial_materials[:4, -3:])

    n_data = (len(vector_param) - mat_count) // len(process_bound)
    process_para = vector_param[mat_count:].reshape(n_data, len(process_bound))

    # Create materials DataFrame indexed by melting temperature
    materials_df = pd.DataFrame(materials_para, columns=params['materials_para_name'])
    materials_df['Melting_T'] = unique_temps

    # Map each dataset to its material parameters
    expanded_materials = []
    for mt in melting_temps:
        row = materials_df[materials_df['Melting_T'] == mt].iloc[0]
        expanded_materials.append(row[params['materials_para_name']].values)

    expanded_materials_df = pd.DataFrame(expanded_materials, columns=params['materials_para_name'])
    process_df = pd.DataFrame(process_para, columns=[params['process_para_name']])

    # Combine results
    result = pd.concat([
        mainfile["Fit_data"].reset_index(drop=True),
        process_condition.reset_index(drop=True),
        process_df.reset_index(drop=True),
        expanded_materials_df.reset_index(drop=True)
    ], axis=1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / "optimized_parameters.csv"
    result.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")

    return result


def main():
    """Main execution function."""
    print(f"Using device: {DEVICE}")
    print(f"Loading configuration from {CONFIG_FILE}")

    # Load configuration
    mainfile = pd.read_csv(CONFIG_FILE)
    alloy_ext = AlloyMaterialDependentExtension(mainfile)
    print(f"Found {len(alloy_ext.unique)} unique materials: {alloy_ext.unique}")

    # Load data
    print("Loading experimental data...")
    fit_data = load_data(mainfile)
    x_data = fit_data["thickness"]
    y_data = fit_data["StressThickness"]

    # Setup parameters
    params = setup_parameters(mainfile)

    # Scale data
    scaler_x = MinMaxScaler(feature_range=(0.1, 1.1))
    scaler_rawy = MinMaxScaler(feature_range=(0.1, 1.1))
    scaler_fity = MinMaxScaler(feature_range=(0.1, 1.1))

    x_scaled = scaler_x.fit_transform(x_data.to_numpy().reshape(-1, 1)).flatten()
    y_scaled = scaler_rawy.fit_transform(y_data.to_numpy().reshape(-1, 1)).flatten()
    scaler_fity.fit(y_data.to_numpy().reshape(-1, 1))

    # Convert to tensors
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(DEVICE)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32).to(DEVICE)
    process_tensor = torch.tensor(params['process_condition'].to_numpy(), dtype=torch.float32).to(DEVICE)
    fit_index_tensor = torch.tensor(fit_data["Index"].to_numpy(), dtype=torch.long).to(DEVICE)

    # Initialize model
    print("Initializing model...")
    model = STFModelTorch(
        x0=params['x_vector_scaled'],
        para_lb=params['para_lb'],
        para_ub=params['para_ub'],
        mainfile=mainfile,
        file_setting=params['file_setting']
    ).to(DEVICE)

    # Train
    model = train_model(model, x_tensor, y_tensor, process_tensor,
                        fit_index_tensor, scaler_x, scaler_fity)

    # Evaluate
    print("Evaluating model...")
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(x_tensor, process_tensor, fit_index_tensor, scaler_x, scaler_fity)

    y_pred = scaler_fity.inverse_transform(y_pred_scaled.cpu().numpy().reshape(-1, 1)).flatten()
    fit_data['y_pred'] = y_pred

    # Plot and save
    print("Generating plots...")
    plot_results(fit_data, params['process_condition'], alloy_ext)

    print("Saving results...")
    save_results(model, params, mainfile, alloy_ext, params['process_condition'])

    print("Done!")


if __name__ == "__main__":
    main()
