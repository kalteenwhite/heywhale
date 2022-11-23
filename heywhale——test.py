import regionmask
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
import xarray as xr
import pandas as pd
import numpy as np
#from cftime import DatetimeNoLeap