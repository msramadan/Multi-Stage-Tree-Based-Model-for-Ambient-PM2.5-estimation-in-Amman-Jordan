from shapely.geometry import Polygon, box
import scipy as sc
from scipy import interpolate
import pandas as pd
import numpy as np
import geopandas as gpd
import rioxarray
import xarray as xr

def fgwi(cdf, regridding_factor, epsilon = 1e-9):

    """
    cdf: NetCDF4 dataset read by xarray.

    regridding_factor: the factor that we want to resample our dataset by; it has to be a multiplier of the original dimension,
    Eg. using a factor of 5 for 0.25 ERA5 will resample the data into 0.05 degree dataset.

    epsilon: a value to avoid floating point error when defining the new dimensions.

    This function resamples the dataset into finer dimension without interpolation, using Kroniker Product.

    Returns xarray dataset with the new spatial dimensions.
    """

    lons = cdf.variables[[name for name in cdf.coords if 'lon' in name.lower()][0]][:]
    lats = cdf.variables[[name for name in cdf.coords if 'lat' in name.lower()][0]][:]
    dif_lon = lons[1] - lons[0]
    dif_lat = lats[0] - lats[1]

    lats_fine = np.float64(np.round(np.arange(lats[0] + (dif_lat/2) - (dif_lat/regridding_factor)/2,
                               lats[-1]- (dif_lat/2) + (dif_lat/regridding_factor)/2,
                               -dif_lat/regridding_factor+epsilon),4))

    lons_fine = np.float64(np.round(np.arange(lons[0] - (dif_lon/2) + (dif_lon/regridding_factor)/2,
                               lons[-1]+ (dif_lon/2) - (dif_lon/regridding_factor)/2,
                               dif_lon/regridding_factor-epsilon),4))

    t = cdf[[name for name in cdf.coords if any(key in name.lower() for key in ['time', 'date', 'year'])][0]]

    # Use the actual coordinate values from the xarray dataset
    times = pd.DatetimeIndex(cdf[t.name].values)

    # Create an empty dictionary to store the resampled data variables
    resampled_vars = {}

    # Iterate through each data variable in the input dataset
    for var_name, data_array in cdf.data_vars.items():
        b=np.array(data_array.values)
        # Adjust var_out shape based on the actual lengths of lats_fine and lons_fine
        var_out = np.zeros((b.shape[0], len(lats_fine), len(lons_fine)))
        for i in range(len(b)):
          mat_in = np.array(b[i])
          x = np.arange(0, mat_in.shape[1], 1)
          y = np.arange(0, mat_in.shape[0], 1)

          # Calculate center of the original grid
          center_x = (x.min() + x.max()) / 2
          center_y = (y.min() + y.max()) / 2

          # Calculate the span of the fine grid in terms of original grid units
          span_x_fine = len(lons_fine) / regridding_factor
          span_y_fine = len(lats_fine) / regridding_factor

          # Define xx and yy centered around the original grid center
          xx = np.linspace(center_x - span_x_fine/2, center_x + span_x_fine/2, len(lons_fine))
          yy = np.linspace(center_y - span_y_fine/2, center_y + span_y_fine/2, len(lats_fine))

          newKernel = interpolate.RectBivariateSpline(y, x, mat_in, kx=3,ky=3) # Note: Spline expects y, x order
          kernelOut = newKernel(yy, xx) # Note: Spline expects yy, xx order

          # Calculate padding
          pad_x = len(lons_fine) - kernelOut.shape[1]
          pad_y = len(lats_fine) - kernelOut.shape[0]

          pad_top = pad_y // 2
          pad_bottom = pad_y - pad_top
          pad_left = pad_x // 2
          pad_right = pad_x - pad_left

          # Apply padding
          padded_kernelOut = np.pad(kernelOut, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')

          var_out[i] = padded_kernelOut


        # Create a DataArray for the resampled variable
        resampled_vars[var_name] = xr.DataArray(
            var_out,
            coords=[times, lats_fine, lons_fine],
            dims=["time", "latitude", "longitude"],
            name=var_name
        )

    # Create a Dataset from the dictionary of resampled data variables
    data_set = xr.Dataset(resampled_vars)

    return data_set