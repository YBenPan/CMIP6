import numpy as np
import pandas as pd
import xarray as xr
import math

area_path = "F:/Computer Programming/Projects/CMIP6/data/fraction/"
area_file = "us_state_gridded.csv"
output_path = "F:/Computer Programming/Projects/CMIP6/data/fraction/"
output_file = "us_state_fraction_0.5x0.5.nc"

lon_start = -179.75
lat_start = -89.75
earth_radius2 = 6.37e6 ** 2
deg2rad = math.pi / 180.0
dx = 0.5
dy = 0.5

# left lon, top lat, state names, shape area
data = pd.read_csv(area_path + area_file, usecols=[1, 2, 10, 14])
states = data["NAME"].values
states = sorted(list(set(states)))  # remove duplicates

# Initialize
fraction = np.zeros((len(states), 360, 720))

for i in range(len(states)):
    state = states[i]
    wk = data[data["NAME"] == state].to_xarray()

    for j in range(len(wk.index)):
        lat = wk["top"][j].values - 0.25
        lon = wk["left"][j].values + 0.25
        lat_ind = int((lat - lat_start) / dy)
        lon_ind = int((lon - lon_start) / dx)

        area = wk["SHAPEAREA"][j].values
        grid_area = (
            earth_radius2 * math.cos(lat * deg2rad) * (dx * deg2rad) * (dy * deg2rad)
        )
        fraction[i, lat_ind, lon_ind] = area / grid_area

fraction[:, :, :] = fraction[:, :, :] / np.nanmax(
    fraction[:, :, :]
)  # almost negligible normalization. max = 1.008

lat_arr = np.arange(-89.75, 90.25, 0.5)
lon_arr = np.arange(-179.75, 180.25, 0.5)

# Output as netCDF
ds = xr.Dataset(
    data_vars=dict(
        fractionState=(["state", "lat", "lon"], fraction),
    ),
    coords=dict(
        state=(["state"], states),
        lat=(["lat"], lat_arr),
        lon=(["lon"], lon_arr),
    ),
    attrs=dict(description="U.S. state fractions on a 0.5x0.5 deg grid"),
)
ds.to_netcdf(output_path + output_file)
ds.close()
