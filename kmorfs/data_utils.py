"""
Data loading and preprocessing utilities for KMORFS.
"""

import numpy as np
import pandas as pd


def RawData_extract(target_col, path_info, plot_setting=0):
    """
    Extract and interpolate raw experimental data from files.

    Parameters
    ----------
    target_col : pd.Series or pd.DataFrame
        Column containing filenames to load. If plot_setting=1, expects
        DataFrame with 'Fit_data' and 'Raw_data' columns.
    path_info : tuple
        (base_path, config_filename, data_folder) paths
    plot_setting : int
        If 1, also loads raw scatter data from Excel files

    Returns
    -------
    tuple
        (Fit_data, Raw_data) DataFrames with interpolated fitting points
        and original scatter data
    """
    if plot_setting:
        target_col, Raw_data_name = target_col["Fit_data"], target_col["Raw_data"]

    Fit_data = None
    Raw_data = None

    for Dataset_index, Dataset_name in enumerate(target_col):
        # Read data file with multiple encoding attempts
        file_path = path_info[0] + path_info[2] + Dataset_name
        temp_data = _read_mixed_encoding(
            file_path,
            sep=r"[\t,]+",
            engine="python",
            skipinitialspace=True
        )

        if plot_setting:
            Raw_Dataset_name = Raw_data_name[Dataset_index]
            temp_scatter_data = pd.read_excel(path_info[0] + path_info[2] + Raw_Dataset_name)
            scatter_data = temp_scatter_data.iloc[:, :2].copy()
            scatter_data.columns = ['thickness', 'StressThickness']
        else:
            scatter_data = temp_data.reset_index(drop=True)

        # Interpolate to adaptive number of points based on thickness range
        temp_data = temp_data.copy()
        thickness_range = temp_data["thickness"].max() - temp_data["thickness"].min()

        # Adaptive point selection
        exponent = 0.7
        scale_factor = 0.44
        max_points = 10
        min_points = 4
        number_of_data = min(max_points, max(min_points,
                                              int(thickness_range ** exponent * scale_factor)))

        interp_thickness = np.linspace(
            temp_data["thickness"].min(),
            temp_data["thickness"].max(),
            number_of_data
        )
        interp_stressthickness = np.interp(
            interp_thickness,
            temp_data["thickness"],
            temp_data["StressThickness"]
        )

        Fitting_data = pd.DataFrame({
            'thickness': interp_thickness,
            'StressThickness': interp_stressthickness
        })
        Fitting_data['Index'] = int(Dataset_index + 1)
        scatter_data['Index'] = int(Dataset_index + 1)

        if Fit_data is None:
            Fit_data = Fitting_data
            Raw_data = scatter_data
        else:
            Fit_data = pd.concat([Fit_data, Fitting_data])
            Raw_data = pd.concat([Raw_data, scatter_data])

    return Fit_data, Raw_data


def _read_mixed_encoding(path, **kwargs):
    """
    Try reading CSV with multiple encodings (utf-8, utf-16, latin-1).
    """
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Unable to decode {path!r} with utf-8, utf-16, or latin-1.")
