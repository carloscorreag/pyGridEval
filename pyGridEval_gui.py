import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import netCDF4 as nc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import pearsonr, spearmanr, skew, kurtosis, ks_2samp, wasserstein_distance, gaussian_kde
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pyproj import Proj, Transformer
import warnings
# Disable all warnings
warnings.filterwarnings('ignore')

# Function to get variable names from the netCDF
def variables_name_nc(f_path):
	# Read the NetCDF file in read mode
	with nc.Dataset(f_path, 'r') as file:
		# Get the variable names from the file
		variable_names = list(file.variables.keys())
	return variable_names

# Function to get the time period
def get_time_period(start_year, end_year):
	return int(start_year), int(end_year)

# Function that extracts 1D latitude and longitude vectors from a NetCDF file with Lambert Conformal Conic projection
def get_lat_lon(dataset):
	
	# Extract coordinates in meters
	x = dataset.variables["x"][:]  # Coordinates on the x-axis (longitude in meters)
	y = dataset.variables["y"][:]  # Coordinates on the y-axis (latitude in meters)

	# Extract Lambert Conformal projection parameters
	proj_params = dataset.variables["Lambert_Conformal"]
	lon_0 = proj_params.longitude_of_central_meridian
	lat_0 = proj_params.latitude_of_projection_origin
	standard_parallel = proj_params.standard_parallel

	# Define the original projection (Lambert Conformal Conic)
	proj_lcc = Proj(proj="lcc", lat_1=standard_parallel, lat_2=standard_parallel,
					lat_0=lat_0, lon_0=lon_0, x_0=proj_params.false_easting,
					y_0=proj_params.false_northing, a=proj_params.earth_radius, b=proj_params.earth_radius)

	# Destination projection (WGS84 in degrees)
	proj_wgs84 = Proj(proj="latlong", datum="WGS84")

	# Create coordinate transformer
	transformer = Transformer.from_proj(proj_lcc, proj_wgs84, always_xy=True)

	# Transform coordinates (keeping 1D dimensions)
	lon, _ = transformer.transform(x, np.zeros_like(x))  # Transform x to longitude
	_, lat = transformer.transform(np.zeros_like(y), y)  # Transform y to latitude

	return np.array(lat), np.array(lon)  # Ensure returned as NumPy arrays


def find_nearest_idx_wrf(lat_target, lon_target, XLAT, XLONG):
	# Calculate the squared difference between coordinates
	dist = np.sqrt((XLAT - lat_target)**2 + (XLONG - lon_target)**2)

	# Find the index of the minimum value in the distance matrix
	min_index = np.unravel_index(np.argmin(dist), dist.shape)

	return min_index



# Function to adjust the surroundings of the point according to the 15 km relaxation condition (Cavalleri et al., 2024) https://doi.org/10.1016/j.atmosres.2024.107734
def relaxation15(grid):
	if grid in [ 'HCLIM_IBERIAxxs_10', 'WRF']:
		c = 9
	elif grid in [ 'HCLIM_IBERIAxxm_40', 'HCLIM_IBERIAxm_40', 'CERRA','CERRA_LAND', 'COSMO-REA6']:
		c = 3
	elif grid in [ 'ERA5','ERA5-Land','EOBS_LR', 'EOBS_HR','CLARA-A3']:
		c = 2
	else:
		c = 2
	return c


# Function to calculate SEEPS (Stable Equitable Error in Probability Space)
def compute_SEEPS(observed, forecast, dry_threshold=0.25, wet_threshold=None, p_clim=None):
	"""
	Calculates the SEEPS index to evaluate precipitation forecast quality.
	"""
	if len(observed) != len(forecast):
		raise ValueError("Las series de observación y pronóstico deben tener la misma longitud.")
	
	def categorize(precip, dry_thr, wet_thr):
		return np.where(precip < dry_thr, 0, np.where(precip < wet_thr, 1, 2))
	
	obs_categories = categorize(observed, dry_threshold, wet_threshold)
	fcst_categories = categorize(forecast, dry_threshold, wet_threshold)
	
	contingency_table = np.zeros((3, 3))
	for obs, fcst in zip(obs_categories, fcst_categories):
		contingency_table[fcst, obs] += 1 #contingency_table[obs, fcst] += 1
	
	total_cases = len(observed)
	#obs_probs = np.sum(contingency_table, axis=1) / total_cases
	#fcst_probs = np.sum(contingency_table, axis=0) / total_cases
	
	if p_clim is None:
		p1 = np.mean(observed < dry_threshold)
	if p1 > 0.85:
		p1 = 0.85
	if p1 < 0.1:
		p1 = 0.1
	
	p2 = (1 - p1) * 2 / 3
	p3 = (1 - p1) * 1 / 3
	
	a11 = 0
	a12 = 1 / (1 - p1)
	a13 = 4 / (1 - p1)
	a21 = 1 / p1
	a22 = 0
	a23 = 3 / (1 - p1)
	a31 = 1 / p1 + 3 / (2 + p1)
	a32 = 3 / (2 + p1)
	a33 = 0
	
	penalty_matrix = 0.5 * np.array([
		[a11, a12, a13],
		[a21, a22, a23],
		[a31, a32, a33]
	])
	
	se = np.sum(contingency_table * penalty_matrix) / total_cases

	'''
	print("\nContingency table (occurrence frequencies):")
	print(contingency_table)
	
	print("\nObserved probabilities:")
	print(obs_probs)
	print("\nForecasted probabilities:")
	print(fcst_probs)
	
	print("\nPenalty matrix:")
	print(penalty_matrix)
	print("\nClimatological probabilities:")
	print(p_clim)
	'''
	return round(se, 3)

# Function to generate metrics and plots
def generate_metrics_and_plots(selected_grids, selected_variable, start_year, end_year, selected_months, interpolation_method):
	print('Selected grids: ' + str(selected_grids))
	print('Selected variable: ' + selected_variable)
	print('Selected interpolation method: ' + interpolation_method)
	print(f'Selected period: {start_year} - {end_year}')
	print(f'Selected months: {selected_months}')

	# Load station data (example file 'stations_data.csv')
	print('loading stations CSV data...')
	try: 
		stations_data_0 = pd.read_csv('stations_data_' + selected_variable + '.csv')
		#stations_data_0 = pd.read_csv('stations_data_' + selected_variable + '.csv')
		stations_data_0['date'] = pd.to_datetime(stations_data_0['date'])
		
		# Filter by the specified period
		stations_data_0 = stations_data_0[(stations_data_0['date'].dt.year >= start_year) & (stations_data_0['date'].dt.year <= end_year)]
		stations_data_0 = stations_data_0[stations_data_0['date'].dt.month.isin(selected_months)]
		def limpiar_valores(x):
			if pd.isna(x):  
				return ""  # If NaN, convert to empty string
			x = str(x)  # Convert everything to string
			if x.endswith(".0"):  
				return x[:-2]  # Remove ".0" only if at the end
			return x  # If not ending in ".0", leave unchanged

		stations_data_0['station_id'] = stations_data_0['station_id'].apply(limpiar_valores)
		print(stations_data_0)
		
		# Convert the 'date' column to datetime type
		stations_data_1 = stations_data_0
		stations_data_1['date'] = pd.to_datetime(stations_data_1['date'])
		
		# Create the complete date range for the desired period
		start_date = f"{start_year}-01-01"
		end_date = f"{end_year}-12-31"
		date_range = pd.date_range(start=start_date, end=end_date)
		date_range = date_range[date_range.month.isin(selected_months)]
		
		# Group by station
		stations = stations_data_1['station_id'].unique()
		results = []
		stations_to_remove = []  
		for station_id in stations:
			
			# Filter station data
			df_station = stations_data_1[stations_data_1['station_id'] == station_id]
			
			# Detect missing days in the DataFrame
			station_date_range = date_range
			missing_days = station_date_range.difference(df_station['date'])
			
			# Detect days with missing values and specific fill values
			df_station['is_nan'] = df_station[selected_variable].isna()
			df_station['is_filled'] = df_station[selected_variable].isin([-99,-99.9, -999,-999.9 -9999, -9999.9])
			nan_days = df_station[df_station['is_nan']]['date']
			filled_days = df_station[df_station['is_filled']]['date']
			
			# Calculate completeness metrics
			total_days = len(station_date_range)
			recorded_days = total_days - len(missing_days) - len(nan_days) - len(filled_days)
			completeness_percentage = (recorded_days / total_days) * 100
			
			# Save completeness results
			results.append({
				"station_id": station_id,
				"latitude": df_station['latitude'].iloc[0],
				"longitude": df_station['longitude'].iloc[0],
				"total_days": total_days,
				"missing_days_count": len(missing_days),
				"nan_days_count": len(nan_days),
				"filled_days_count": len(filled_days),
				"days_with_valid_data": recorded_days,
				"days_without_valid_data": total_days - recorded_days,
				"completeness_percentage": completeness_percentage
			})
		
			# If station has less than 90% completeness, add it to the removal list
			if completeness_percentage < 90:
				stations_to_remove.append(station_id)
		
		# Convert results to DataFrame
		results_df= pd.DataFrame(results)

		# Categorize stations by completeness level
		bins = [0, 10, 50, 90, 99, 100]
		labels = ["<10%", "10-50%", "50-90%", "90-99%", "99-100%"]
		results_df['completeness_category'] = pd.cut(results_df['completeness_percentage'], bins=bins, labels=labels)
		# fix stations with 0% completeness
		results_df.loc[results_df['completeness_percentage'] == 0, 'completeness_category'] = "<10%"
		
		# Save to CSV file
		results_df.to_csv("stations_completeness.csv", index=False)
		print('stations_completeness.csv has been saved')
		
		# Plot the map
		plt.figure(figsize=(10, 8))
		# Create the axis with a projection
		ax = plt.axes(projection=ccrs.PlateCarree())
		# Add coastlines and borders
		ax.coastlines(resolution='10m', linewidth=1)  # Coastlines
		ax.add_feature(cfeature.BORDERS, linestyle='--')
		# Plot the data
		scatter = ax.scatter(
			results_df['longitude'], results_df['latitude'],
			c=results_df['completeness_percentage'], cmap='viridis', s=100, edgecolor='k', vmin=0, vmax=100,
			transform=ccrs.PlateCarree()  # Transformación a coordenadas geográficas
		)
		# Add colorbar
		colorbar = plt.colorbar(scatter, ax=ax, label="completeness percentage (%)")
		colorbar.set_ticks([0, 20, 40, 60, 80, 100])
		# Add title and labels
		plt.title("Stations completeness percentage")
		plt.xlabel("Longitude")
		plt.ylabel("Latitude")
		# Save the file
		plt.savefig("stations_completeness_map.png")
		print('stations_completeness_map.png has been saved')
		# Close the figure
		plt.close()
		
		# Count stations by category
		summary = results_df['completeness_category'].value_counts().sort_index()
		total_stations = len(results_df)
		print("completeness summary:")
		for label, count in summary.items():
			percentage = (count / total_stations) * 100
			print(f"{label}: {count} stations ({percentage:.2f}%)")
		
		del stations_data_1
		del results
		del results_df
		
		# Filter rows by removing those with NaN
		stations_data_0 = stations_data_0.dropna()
		# Filter rows by removing those with -99, -999, or -9999
		values_to_remove = [-99,-99.9, -999,-999.9 -9999, -9999.9]
		stations_data_0 = stations_data_0[~stations_data_0[selected_variable].isin(values_to_remove)]

		# Configuration: enable or disable removal of stations with completeness < 90%
		remove_incomplete_stations = True

		if remove_incomplete_stations:
			# Number of stations before removal
			stations_before = stations_data_0['station_id'].nunique()
			# Remove stations with less than 90% completeness
			stations_data_0 = stations_data_0[~stations_data_0['station_id'].isin(stations_to_remove)]
			# Number of stations after removal
			stations_after = stations_data_0['station_id'].nunique()
			# Show how many stations were removed
			print(f"{len(stations_to_remove)} stations with less than 90% completeness have been removed.")
			# Check if no valid stations remain
			if stations_after == 0:
				raise ValueError("No valid stations remain after applying the completeness filter.")
		else:
			# Message if completeness filter is not applied
			print("No stations were removed (completeness filter disabled).")

	except FileNotFoundError:
		print(f'Error - file not found: stations_data_{selected_variable}.csv')
		exit()

	for grid in selected_grids:
		stations_data = stations_data_0 # create a new DataFrame containing the station CSV data
		# Load grid data ('HCLIM_IBERIAxm_40', 'HCLIM_IBERIAxxm_40', 'HCLIM_IBERIAxxs_10', 'ISIMIP-CHELSA', 'CHIRTS', 'CHIRPS', 'ERA5', 'ERA5-Land', 'COSMO-REA6', 'CERRA', 'CERRA_LAND', 'EOBS', 'EOBS_HR', 'EOBS_LR') using netCDF4
		print(grid)
		print('loading netCDF grid data...')

		try:
			
			grid_file = 'grid_data_' + grid + '_' + selected_variable + '.nc'
			#grid_file = 'grid_data_' + grid + '_' + selected_variable + '.nc'
			grid_data = nc.Dataset(grid_file)

		except:
			print('Error - file not found: grid_data_' + grid + '_' + selected_variable + '.nc')
			exit()

		# Define variable names
		names = variables_name_nc(grid_file)
		
		
		if grid in ['ERA5-Land'] and selected_variable in ['precipitation']:
			targetvar = [string for string in names if string in ['tp']][0]
		if grid in ['EOBS_HR'] and selected_variable in ['precipitation']:
			targetvar = [string for string in names if string in ['rr']][0]
		if grid in ['EOBS_LR'] and selected_variable in ['precipitation']:
			targetvar = [string for string in names if string in ['rr']][0]
		if grid in ['ERA5'] and selected_variable in ['precipitation']:
			targetvar = [string for string in names if string in ['tp']][0]
		if grid in ['CERRA'] and selected_variable in ['precipitation']:
			targetvar = [string for string in names if string in ['var61']][0]	
		if grid == 'COSMO-REA6' and selected_variable == 'wind_speed':
    			targetvar = [string for string in names if string == 'var33'][0]
		if grid == 'COSMO-REA6' and selected_variable == 'humidity':
			targetvar = [string for string in names if string == 'var52'][0]
		elif grid in ['ERA5-Land'] and selected_variable == 'wind_speed':
    			targetvar = [string for string in names if string == 'u10'][0]
    			if not targetvar:
        			raise ValueError(f"Not found wind variables ('u10', 'v10') en: {names}")
		elif grid in ['ERA5-Land'] and selected_variable == 'humidity':
			targetvar = [string for string in names if string == 'hr2m'][0]
			if not targetvar:
				raise ValueError(f"Not found humidity variables: {names}")
		elif grid in ['ERA5-Land'] and selected_variable not in ['precipitation', 'wind_speed', 'humidity']:
    			targetvar = [string for string in names if string in ['t2m']][0]
		else:
			targetvar = names[-1]
		# Select target variable based on grid and selected variable
		# (e.g., precipitation, wind_speed, humidity, temperature)
		# Select latitude, longitude, and time variable names
		targetlat = [string for string in names if 'lat' in string.lower()][0]
		targetlon = [string for string in names if 'lon' in string.lower()][0]
		targettime = [string for string in names if string in ['time','XTIME', 'valid_time']][0]

		# Variables inside the netCDF file and units conversion
		if grid not in [ 'HCLIM_IBERIAxxs_10', 'HCLIM_IBERIAxxm_40', 'HCLIM_IBERIAxm_40']: 
			grid_lat = grid_data.variables[targetlat][:]
			grid_lon = grid_data.variables[targetlon][:]
		else: 
			# HCLIM is not on regular lat/lon grid, conversion is needed
			grid_lat, grid_lon = get_lat_lon(grid_data)
		grid_time = grid_data.variables[targettime][:]

		# Handle units of time variable
		try:
			if not np.any(grid_time.mask) == True:
				units = grid_data.variables[targettime].units
			else:
				units = 'days since 1991-01-01 00:00:00'
				grid_time = np.array(list(range(grid_time.shape[0])))
		except:
			if grid == 'CHIRTS':
				units = 'days since 1980-01-01 00:00:00'
			else:
				print('Error - units not found in netCDF metadata')
		
		if selected_variable in ['temperature', 'maximum_temperature', 'minimum_temperature'] and grid not in ['CHIRTS', 'ERA5', 'COSMO-REA6']:
			grid_targetvar = grid_data.variables[targetvar][:].astype('float32') - 273.15 # convert from Kelvin to Celsius
		elif selected_variable in ['temperature', 'maximum_temperature', 'minimum_temperature'] and grid in ['CHIRTS', 'ERA5']:
			grid_targetvar = grid_data.variables[targetvar][:].astype('float32') # keep units in Celsius
		elif selected_variable in ['temperature', 'maximum_temperature', 'minimum_temperature'] and grid in ['COSMO-REA6']:
			grid_targetvar = grid_data.variables[targetvar][:][:,0,:,:].astype('float32') # keep units in Celsius
		elif selected_variable in ['wind_speed'] and grid not in ['COSMO-REA6']:
			grid_targetvar = grid_data.variables[targetvar][:].astype('float32') # keep wind units
		elif selected_variable in ['wind_speed'] and grid in ['COSMO-REA6']:
			grid_targetvar = grid_data.variables[targetvar][:][:,0,:,:].astype('float32') # keep wind units and convert from 4D to 3D
		elif selected_variable == 'precipitation' and grid not in ['ISIMIP-CHELSA', 'CHIRPS', 'CERRA', 'HCLIM_IBERIAxxs_10', 'HCLIM_IBERIAxxm_40', 'HCLIM_IBERIAxm_40', 'WRF']:
			grid_targetvar = grid_data.variables[targetvar][:].astype('float32')*1000 # convert m/day to mm/day
		elif selected_variable == 'precipitation' and grid == 'CERRA':
			grid_targetvar = grid_data.variables[targetvar][:][:,0,:,:].astype('float32')*1000 # convert m/day to mm/day and convert from 4D to 3D
		elif selected_variable == 'precipitation' and grid in ['CHIRPS', 'WRF']:
			grid_targetvar = grid_data.variables[targetvar][:].astype('float32') # keep units in mm/day
		elif selected_variable == 'precipitation' and grid in ['ISIMIP-CHELSA', 'HCLIM_IBERIAxxs_10', 'HCLIM_IBERIAxxm_40', 'HCLIM_IBERIAxm_40']:
			grid_targetvar = grid_data.variables[targetvar][:].astype('float32')*86400 # convert kg/m²s to mm/day
		elif selected_variable in ['humidity'] and grid not in ['COSMO-REA6']:
			grid_targetvar = grid_data.variables[targetvar][:].astype('float32') # keep humidity units
		elif selected_variable in ['humidity'] and grid in ['COSMO-REA6']:
			grid_targetvar = grid_data.variables[targetvar][:][:,0,:,:].astype('float32') # keep humidity units and convert from 4D to 3D
		elif selected_variable in ['radiation'] and grid not in ['COSMO-REA6']:
			grid_targetvar = grid_data.variables[targetvar][:].astype('float32') # keep radiation units
		elif selected_variable in ['radiation'] and grid in ['COSMO-REA6']:
			grid_targetvar = grid_data.variables[targetvar][:][:,0,:,:].astype('float32') # keep radiation units and convert from 4D to 3D
		else:
			print('Error - units')
			exit()

		if grid != 'WRF':
			# force ascending order
			if (grid_lat[0] > grid_lat[-1]):
				grid_lat = np.flip(grid_lat)
				grid_targetvar = np.flip(grid_targetvar, axis=1)
			if (grid_lon[0] > grid_lon[-1]):
				grid_lon = np.flip(grid_lon)
				grid_targetvar = np.flip(grid_targetvar, axis=2)

		# Convert variable units if needed (Kelvin to Celsius, m/day to mm/day, etc.)
		# Handle 4D to 3D conversion for specific grids (COSMO-REA6, CERRA)
		# Flip latitude and longitude arrays if not in ascending order
		# Handle masks, NaNs, and fill values
		grid_targetvar = np.where(grid_targetvar.mask, np.nan, grid_targetvar) # replace masked values with NaN
		fill_value = getattr(grid_data.variables[targetvar], '_FillValue', None)  # detect _FillValue
		grid_targetvar = np.where(grid_targetvar == fill_value, np.nan, grid_targetvar) # replace _FillValue with NaN
		grid_data.close() # close the netCDF
		del grid_data

		# Function to convert dates to time indices in the grid
		def convert_time_to_index(time_array, date):
			# Convert target date to a number in original units
			time_num = nc.date2num(date, units=units, calendar='standard')
			# Calculate difference in units (same unit system)
			time_diff = np.abs(time_array - time_num)
			# Check if any difference is greater than 24 hours, in compatible units
			if units.startswith('days'):
				threshold = 1  # 1 day
			elif units.startswith('minutes'):
				threshold = 60 * 24  # 24 hours
			elif units.startswith('hours'):
				threshold = 24  # 24 hours
			elif units.startswith('seconds'):
				threshold = 24 * 3600  # 24 hours * 3600 seconds
			else:
				raise ValueError("Unsupported units in calculation.")
			# If difference exceeds threshold, return NaN
			if np.all(time_diff > threshold):
				return np.nan
			# Interpolate if differences are less than or equal to 24 hours
			time_idx = np.interp(time_num, time_array, np.arange(len(time_array)))

			if grid in ['CERRA','CERRA_LAND'] and selected_variable == 'precipitation':
				return int(time_idx)
			else:
				return round(time_idx)

			

		# Create the interpolator for the grid data
		def create_interpolator(targetvar_data, lat_array, lon_array, lat_station, lon_station, time_idx, interpolation_method, date):
			
			if grid != 'WRF':
				lat_array = np.sort(np.unique(lat_array))
				lon_array = np.sort(np.unique(lon_array))
				lat_idx = np.abs(lat_array - lat_station).argmin()
				lon_idx = np.abs(lon_array - lon_station).argmin()
				
				# Define ranges for local interpolation
				if interpolation_method == '15km relaxation':
					lat_range = lat_array[max(0, lat_idx-(relaxation15(grid)-1)):min(len(lat_array), lat_idx+relaxation15(grid))]
					lon_range = lon_array[max(0, lon_idx-(relaxation15(grid)-1)):min(len(lon_array), lon_idx+relaxation15(grid))]
					targetvar_range = targetvar_data[time_idx, max(0, lat_idx-(relaxation15(grid)-1)):min(len(lat_array), lat_idx+relaxation15(grid)), max(0, lon_idx-(relaxation15(grid)-1)):min(len(lon_array), lon_idx+relaxation15(grid))]
				else:
					lat_range = lat_array[max(0, lat_idx-1):min(len(lat_array), lat_idx+2)]
					lon_range = lon_array[max(0, lon_idx-1):min(len(lon_array), lon_idx+2)]    
					targetvar_range = targetvar_data[time_idx, max(0, lat_idx-1):min(len(lat_array), lat_idx+2), max(0, lon_idx-1):min(len(lon_array), lon_idx+2)]
			else:
				idx_wrf = find_nearest_idx_wrf(lat_station, lon_station, lat_array, lon_array)
				lat_idx = idx_wrf[0]
				lon_idx = idx_wrf[1]
				
				# Define ranges for local interpolation
				if interpolation_method == '15km relaxation':
					lat_range = lat_array[max(0, lat_idx-(relaxation15(grid)-1)):min(len(lat_array[:,0]), lat_idx+relaxation15(grid)), lon_idx]  
					lon_range = lon_array[lat_idx, max(0, lon_idx-(relaxation15(grid)-1)):min(len(lon_array[0,:]), lon_idx+relaxation15(grid))]
					targetvar_range = targetvar_data[time_idx, max(0, lat_idx-(relaxation15(grid)-1)):min(len(lat_array[:,0]), lat_idx+relaxation15(grid)), max(0, lon_idx-(relaxation15(grid)-1)):min(len(lon_array[0,:]), lon_idx+relaxation15(grid))]
				else:
					lat_range = lat_array[max(0, lat_idx-1):min(len(lat_array[:,0]), lat_idx+2), lon_idx]  
					lon_range = lon_array[lat_idx, max(0, lon_idx-1):min(len(lon_array[0,:]), lon_idx+2)]
					targetvar_range = targetvar_data[time_idx, max(0, lat_idx-1):min(len(lat_array[:,0]), lat_idx+2), max(0, lon_idx-1):min(len(lon_array[0,:]), lon_idx+2)]
					
			# Check and handle `np.nan` in targetvar_range. If all are `np.nan`, return `np.nan`
			if np.isnan(targetvar_range).all():
				targetvar_range[:] = np.nan  # e.g., fill everything with NaNs
				
			# Check and handle `np.nan` in targetvar_range. If some are NaN, fill with the nearest valid value to the center
			if np.isnan(targetvar_range).any() and not np.isnan(targetvar_range).all():
				center_idx = (targetvar_range.shape[0] // 2, targetvar_range.shape[1] // 2)  # approximate center
				valid_mask = ~np.isnan(targetvar_range)  # mask of valid values
				valid_indices = np.argwhere(valid_mask)  # indices of valid values
				
				# Distance to center from valid values
				distances = np.linalg.norm(valid_indices - np.array(center_idx), axis=1)
				closest_valid_idx = valid_indices[distances.argmin()]  # closest valid value index
				
				# Fill NaN with the closest valid value
				targetvar_range[~valid_mask] = targetvar_range[tuple(closest_valid_idx)]

			if interpolation_method == '15km relaxation' and grid not in ['ERA5', 'EOBS_LR', 'CLARA-A3']:
				stations_data_filter = stations_data[(stations_data['latitude'] == lat_station) & (stations_data['longitude'] == lon_station) & (stations_data['date'] == date)]
				stations_data_value = stations_data_filter[selected_variable].iloc[0]
				# Select the value closest to the station value
				closest_idx_value = np.unravel_index(np.abs(targetvar_range - stations_data_value).argmin(), targetvar_range.shape)
				closest_value = targetvar_range[closest_idx_value]

				return closest_value
			elif interpolation_method == '15km relaxation' and grid in ['ERA5', 'EOBS_LR','CLARA-A3']:
				return RegularGridInterpolator(
					(lat_range, lon_range),
					targetvar_range,
					method='nearest',
					bounds_error=True,
					fill_value=np.nan
				)
			else: 
				return RegularGridInterpolator(
					(lat_range, lon_range),
					targetvar_range,
					method=interpolation_method,
					bounds_error=True,
					fill_value=np.nan
				)

		# Function to extract interpolated grid values for station locations and dates
		def extract_interpolated_grid_value(lat, lon, date):
			time_idx = convert_time_to_index(grid_time, date)
			try: 
				if interpolation_method == '15km relaxation' and grid not in ['ERA5', 'EOBS_LR','CLARA-A3']:
					interpolated_value  = create_interpolator(grid_targetvar, grid_lat, grid_lon, lat, lon, time_idx,interpolation_method, date)
					if selected_variable == 'precipitation' and interpolated_value < 0:
						interpolated_value = 0
					return interpolated_value
				else:
					interpolator = create_interpolator(grid_targetvar, grid_lat, grid_lon, lat, lon, time_idx,interpolation_method, date)
					interpolated_value = interpolator((lat, lon)) 
					if selected_variable == 'precipitation' and interpolated_value < 0:
						interpolated_value = 0
					return interpolated_value
			except:
				return np.nan    

			print('data loaded')

		# Apply extraction to each row of the stations DataFrame using local interpolation
		print('interpolating...')
		print('please wait')
		stations_data['interpolated_grid_value'] = stations_data.apply(
			lambda row: extract_interpolated_grid_value(row['latitude'], row['longitude'], row['date']),
			axis=1
		)

		# Calculate differences and metrics
		stations_data['interpolated_grid_value'] = stations_data['interpolated_grid_value'].apply(lambda x: x.filled(np.nan) if isinstance(x, np.ma.MaskedArray) else x) 
		stations_data = stations_data.dropna() 
		#print(stations_data[~stations_data['interpolated_grid_value'].apply(lambda x: isinstance(x, (int, float)))])
		#print(stations_data[~stations_data[selected_variable].apply(lambda x: isinstance(x, (int, float)))])
		stations_data['interpolated_grid_value'] = pd.to_numeric(stations_data['interpolated_grid_value'], errors='coerce') 
		stations_data[selected_variable] = pd.to_numeric(stations_data[selected_variable], errors='coerce')
		stations_data.to_csv(f'{selected_variable}_stations_data_and_interpolated_grid_value_{grid}.csv', index=False)
		print('interpolation completed')
		print('obtaining metrics...')
		print('please wait')
		stations_data_accu_ini = stations_data
		stations_data['difference_interpolated'] = stations_data['interpolated_grid_value'] - stations_data[selected_variable] 
		
		def calculate_metrics_interpolated_temp(data):
			data = data.dropna() 
			if len(data) < 2:
				return pd.Series({
					'Mean Bias': np.nan,
					'Mean Absolute Error': np.nan,
					'RMSE': np.nan,
					'Correlation': np.nan,
					'Variance Bias': np.nan
				})
			mean_bias = data['difference_interpolated'].mean()
			mean_absolute_error = data['difference_interpolated'].abs().mean()
			rmse = np.sqrt((data['difference_interpolated'] ** 2).mean())
			correlation, _ = pearsonr(data['interpolated_grid_value'], data[selected_variable])
			variance_bias = data['interpolated_grid_value'].var() - data[selected_variable].var()
			percentile90_bias = np.percentile(data['interpolated_grid_value'], 90) - np.percentile(data[selected_variable], 90)
			percentile10_bias = np.percentile(data['interpolated_grid_value'], 10) - np.percentile(data[selected_variable], 10)
			std_bias = data['interpolated_grid_value'].std() - data[selected_variable].std()
			wd = wasserstein_distance(data['interpolated_grid_value'], data[selected_variable])
			ks_stat, ks_p = ks_2samp(data['interpolated_grid_value'], data[selected_variable])
			skew_bias = skew(data['interpolated_grid_value']) - skew(data[selected_variable])
			kurtosis_bias = kurtosis(data['interpolated_grid_value']) - kurtosis(data[selected_variable])
			
			return pd.Series({
				'Mean Bias': mean_bias,
				'P90 Bias': percentile90_bias,
				'P10 Bias': percentile10_bias,
				'Mean Absolute Error': mean_absolute_error,
				'RMSE': rmse,
				'Correlation': correlation,
				'Variance Bias': variance_bias,
				'Std Bias': std_bias,
				'Wasserstein Distance': wd,
				'KS test stat': ks_stat,
				'KS test p': ks_p,
				'Skew Bias': skew_bias,
				'Kurtosis Bias': kurtosis_bias
			})

		def calculate_metrics_interpolated_wspeed(data):
			data = data.dropna()  
			if len(data) < 2:
				return pd.Series({
					'Mean Bias': np.nan,
					'Mean Absolute Error': np.nan,
					'RMSE': np.nan,
					'Correlation': np.nan,
					'Variance Bias': np.nan
				})
			mean_bias = data['difference_interpolated'].mean()
			mean_absolute_error = data['difference_interpolated'].abs().mean()
			rmse = np.sqrt((data['difference_interpolated'] ** 2).mean())
			correlation, _ = pearsonr(data['interpolated_grid_value'], data[selected_variable])
			variance_bias = data['interpolated_grid_value'].var() - data[selected_variable].var()
			percentile90_bias = np.percentile(data['interpolated_grid_value'], 90) - np.percentile(data[selected_variable], 90)
			percentile10_bias = np.percentile(data['interpolated_grid_value'], 10) - np.percentile(data[selected_variable], 10)
			percentile95_bias = np.percentile(data['interpolated_grid_value'], 95) - np.percentile(data[selected_variable], 95)
			std_bias = data['interpolated_grid_value'].std() - data[selected_variable].std()
			wd = wasserstein_distance(data['interpolated_grid_value'], data[selected_variable])
			ks_stat, ks_p = ks_2samp(data['interpolated_grid_value'], data[selected_variable])
			skew_bias = skew(data['interpolated_grid_value']) - skew(data[selected_variable])
			kurtosis_bias = kurtosis(data['interpolated_grid_value']) - kurtosis(data[selected_variable])
			
			return pd.Series({
				'Mean Bias': mean_bias,
				'P95 Bias': percentile95_bias,
				'P90 Bias': percentile90_bias,
				'P10 Bias': percentile10_bias,
				'Mean Absolute Error': mean_absolute_error,
				'RMSE': rmse,
				'Correlation': correlation,
				'Variance Bias': variance_bias,
				'Std Bias': std_bias,
				'Wasserstein Distance': wd,
				'KS test stat': ks_stat,
				'KS test p': ks_p,
				'Skew Bias': skew_bias,
				'Kurtosis Bias': kurtosis_bias
			})
		
		def calculate_metrics_interpolated_radiation(data):
			data = data.dropna()  
			if len(data) < 2:
				return pd.Series({
					'Mean Bias': np.nan,
					'Mean Absolute Error': np.nan,
					'RMSE': np.nan,
					'Correlation': np.nan,
					'Variance Bias': np.nan
				})
			mean_bias = data['difference_interpolated'].mean()
			mean_absolute_error = data['difference_interpolated'].abs().mean()
			rmse = np.sqrt((data['difference_interpolated'] ** 2).mean())
			correlation, _ = pearsonr(data['interpolated_grid_value'], data[selected_variable])
			variance_bias = data['interpolated_grid_value'].var() - data[selected_variable].var()
			percentile90_bias = np.percentile(data['interpolated_grid_value'], 90) - np.percentile(data[selected_variable], 90)
			percentile10_bias = np.percentile(data['interpolated_grid_value'], 10) - np.percentile(data[selected_variable], 10)
			percentile95_bias = np.percentile(data['interpolated_grid_value'], 95) - np.percentile(data[selected_variable], 95)
			std_bias = data['interpolated_grid_value'].std() - data[selected_variable].std()
			wd = wasserstein_distance(data['interpolated_grid_value'], data[selected_variable])
			ks_stat, ks_p = ks_2samp(data['interpolated_grid_value'], data[selected_variable])
			skew_bias = skew(data['interpolated_grid_value']) - skew(data[selected_variable])
			kurtosis_bias = kurtosis(data['interpolated_grid_value']) - kurtosis(data[selected_variable])

			
			return pd.Series({
				'Mean Bias': mean_bias,
				'P95 Bias': percentile95_bias,
				'P90 Bias': percentile90_bias,
				'P10 Bias': percentile10_bias,
				'Mean Absolute Error': mean_absolute_error,
				'RMSE': rmse,
				'Correlation': correlation,
				'Variance Bias': variance_bias,
				'Std Bias': std_bias,
				'Wasserstein Distance': wd,
				'KS test stat': ks_stat,
				'KS test p': ks_p,
				'Skew Bias': skew_bias,
				'Kurtosis Bias': kurtosis_bias
			})
		
		def calculate_metrics_interpolated_precip(data):
			data = data.dropna() 
			if len(data) < 2:
				return pd.Series({
					'Mean Bias': np.nan,
					'Mean Absolute Error': np.nan,
					'Correlation': np.nan,
					'Variance Bias': np.nan
				})
			mean_bias = data['difference_interpolated'].mean() # data['interpolated_grid_value'].mean() - stations_data[selected_variable].mean()
			if stations_data[selected_variable].mean() < 0.001 and data['interpolated_grid_value'].mean() > 0.001:
				mean_relative_bias = np.nan
			elif data[selected_variable].mean() < 0.001 and data['interpolated_grid_value'].mean() < 0.001:
				mean_relative_bias = 0
			else:
				mean_relative_bias = 100 * (data['interpolated_grid_value'].mean() - stations_data[selected_variable].mean())/ stations_data[selected_variable].mean()
				
			multiplicative_bias = (data['interpolated_grid_value'].mean() / stations_data[selected_variable].mean())
			mean_absolute_error = data['difference_interpolated'].abs().mean()
			correlation, _ = spearmanr(data['interpolated_grid_value'], data[selected_variable])
			variance_bias = data['interpolated_grid_value'].var() - data[selected_variable].var()
			std_bias = data['interpolated_grid_value'].std() - data[selected_variable].std()
			
			if data[selected_variable].var() < 0.001 and data['interpolated_grid_value'].var() > 0.001:
				var_relative_bias = np.nan
			elif data[selected_variable].var() < 0.001 and data['interpolated_grid_value'].var() < 0.001:
				var_relative_bias = 0
			else:
				var_relative_bias = 100 * (data['interpolated_grid_value'].var() - data[selected_variable].var()) / data[selected_variable].var()
				
			if data[selected_variable].std() < 0.001 and data['interpolated_grid_value'].std() > 0.001:
				std_relative_bias = np.nan
			elif data[selected_variable].std() < 0.001 and data['interpolated_grid_value'].std() < 0.001:
				std_relative_bias = 0
			else:
				std_relative_bias = 100 * (data['interpolated_grid_value'].std() - data[selected_variable].std()) / data[selected_variable].std()
			
			percentile10_bias = np.percentile(data['interpolated_grid_value'], 10) - np.percentile(data[selected_variable], 10)
			percentile90_bias = np.percentile(data['interpolated_grid_value'], 90) - np.percentile(data[selected_variable], 90)
			percentile95_bias = np.percentile(data['interpolated_grid_value'], 95) - np.percentile(data[selected_variable], 95)
			percentile99_bias = np.percentile(data['interpolated_grid_value'], 99) - np.percentile(data[selected_variable], 99)
			
			if np.percentile(data[selected_variable], 99) < 0.001 and np.percentile(data['interpolated_grid_value'], 99) > 0.001:
				percentile99_relative_bias = np.nan
			elif np.percentile(data[selected_variable], 99) < 0.001 and np.percentile(data['interpolated_grid_value'], 99) < 0.001:
				percentile99_relative_bias = 0
			else:
				percentile99_relative_bias = 100 * (np.percentile(data['interpolated_grid_value'], 99) - np.percentile(data[selected_variable], 99)) / np.percentile(data[selected_variable], 99)
			
			if np.percentile(data[selected_variable], 95) < 0.001 and np.percentile(data['interpolated_grid_value'], 95) > 0.001:
				percentile95_relative_bias = np.nan
			elif np.percentile(data[selected_variable], 95) < 0.001 and np.percentile(data['interpolated_grid_value'], 95) < 0.001:
				percentile95_relative_bias = 0
			else:
				percentile95_relative_bias = 100 * (np.percentile(data['interpolated_grid_value'], 95) - np.percentile(data[selected_variable], 95)) / np.percentile(data[selected_variable], 95)
			
			if np.percentile(data[selected_variable], 90) < 0.001 and np.percentile(data['interpolated_grid_value'], 90) > 0.001:
				percentile90_relative_bias = np.nan
			elif np.percentile(data[selected_variable], 90) < 0.001 and np.percentile(data['interpolated_grid_value'], 90) < 0.001:
				percentile90_relative_bias = 0
			else:
				percentile90_relative_bias = 100 * (np.percentile(data['interpolated_grid_value'], 90) - np.percentile(data[selected_variable], 90)) / np.percentile(data[selected_variable], 90)
			
			if np.percentile(data[selected_variable], 10) < 0.001 and np.percentile(data['interpolated_grid_value'], 10) > 0.001:
				percentile10_relative_bias = np.nan
			elif np.percentile(data[selected_variable], 10) < 0.001 and np.percentile(data['interpolated_grid_value'], 10) < 0.001:
				percentile10_relative_bias = 0
			else:
				percentile10_relative_bias = 100 * (np.percentile(data['interpolated_grid_value'], 10) - np.percentile(data[selected_variable], 10)) / np.percentile(data[selected_variable], 10)
			
			wd = wasserstein_distance(data['interpolated_grid_value'], data[selected_variable])
			ks_stat, ks_p = ks_2samp(data['interpolated_grid_value'], data[selected_variable])
			skew_bias = skew(data['interpolated_grid_value']) - skew(data[selected_variable])
			kurtosis_bias = kurtosis(data['interpolated_grid_value']) - kurtosis(data[selected_variable])
			
			# Filter only the rainy days
			data_rain = data[data[selected_variable] > 0.25]
			# Function to calculate the precipitation threshold to split in a 2:1 ratio
			def calculate_threshold(group):
				sorted_precip = group[selected_variable].sort_values()
				n = len(sorted_precip)
				index_threshold = n * 2 // 3
				return sorted_precip.iloc[index_threshold]
			
			# Calculate the precipitation threshold to split in a 2:1 ratio
			sorted_precip = data_rain[selected_variable].dropna().sort_values() 

			n = len(sorted_precip)  

			if n == 0:
				# print(f"Warning: sorted_precip is empty for variable {selected_variable}. A default value will be assigned.")
				wet_threshold = np.nan  # O puedes usar un valor predeterminado como 0
			elif n == 1:
				# print(f"Warning: only one value in sorted_precip. Cannot calculate a meaningful threshold.")
				wet_threshold = sorted_precip.iloc[0]  # Único valor disponible
			else:
				index_threshold = min(n * 2 // 3, n - 1)  # Ensure the index is within bounds
				wet_threshold = sorted_precip.iloc[index_threshold]

			# Calculate SEEPS using the computed threshold
			if np.isnan(wet_threshold):  
				seeps_skill_score = np.nan  
			else:
				seeps_skill_score = 1 - compute_SEEPS(
					data[selected_variable].values, 
					data["interpolated_grid_value"].values, 
					wet_threshold=wet_threshold
				)
			
			return pd.Series({
				'Mean Bias': mean_bias,
				'Mean relative Bias': mean_relative_bias,
				'Mean Absolute Error': mean_absolute_error,
				'Multiplicative Bias': multiplicative_bias,
				'P99 Bias': percentile99_bias,
				'P99 relative Bias': percentile99_relative_bias,
				'P95 Bias': percentile95_bias,
				'P95 relative Bias': percentile95_relative_bias,
				'P90 Bias': percentile90_bias,
				'P90 relative Bias': percentile90_relative_bias,
				'P10 Bias': percentile10_bias,
				'P10 relative Bias': percentile10_relative_bias,
				'Correlation': correlation,
				'SEEPS Skill Score': seeps_skill_score,
				'Variance Bias': variance_bias,
				'Variance relative Bias': var_relative_bias,
				'Std Bias': std_bias,
				'Std relative Bias': std_relative_bias,
				'Wasserstein Distance': wd,
				'KS test stat': ks_stat,
				'KS test p': ks_p,
				'Skew Bias': skew_bias,
				'Kurtosis Bias': kurtosis_bias
			})
			
		def calculate_metrics_interpolated_humidity(data):
			data = data.dropna()  
			if len(data) < 2:
				return pd.Series({
					'Mean Bias': np.nan,
					'Mean Absolute Error': np.nan,
					'RMSE': np.nan,
					'Correlation': np.nan,
					'Variance Bias': np.nan
				})
			mean_bias = data['difference_interpolated'].mean()
			mean_absolute_error = data['difference_interpolated'].abs().mean()
			rmse = np.sqrt((data['difference_interpolated'] ** 2).mean())
			correlation, _ = pearsonr(data['interpolated_grid_value'], data[selected_variable])
			variance_bias = data['interpolated_grid_value'].var() - data[selected_variable].var()
			percentile90_bias = np.percentile(data['interpolated_grid_value'], 90) - np.percentile(data[selected_variable], 90)
			percentile10_bias = np.percentile(data['interpolated_grid_value'], 10) - np.percentile(data[selected_variable], 10)
			percentile95_bias = np.percentile(data['interpolated_grid_value'], 95) - np.percentile(data[selected_variable], 95)
			std_bias = data['interpolated_grid_value'].std() - data[selected_variable].std()
			wd = wasserstein_distance(data['interpolated_grid_value'], data[selected_variable])
			ks_stat, ks_p = ks_2samp(data['interpolated_grid_value'], data[selected_variable])
			skew_bias = skew(data['interpolated_grid_value']) - skew(data[selected_variable])
			kurtosis_bias = kurtosis(data['interpolated_grid_value']) - kurtosis(data[selected_variable])
			
			return pd.Series({
				'Mean Bias': mean_bias,
				'P95 Bias': percentile95_bias,
				'P90 Bias': percentile90_bias,
				'P10 Bias': percentile10_bias,
				'Mean Absolute Error': mean_absolute_error,
				'RMSE': rmse,
				'Correlation': correlation,
				'Variance Bias': variance_bias,
				'Std Bias': std_bias,
				'Wasserstein Distance': wd,
				'KS test stat': ks_stat,
				'KS test p': ks_p,
				'Skew Bias': skew_bias,
				'Kurtosis Bias': kurtosis_bias
			})
		
		# Calculate metrics based on the selected variable
		if selected_variable in ['temperature', 'maximum_temperature', 'minimum_temperature']:
			metrics_per_station_interpolated = stations_data.groupby('station_id').apply(calculate_metrics_interpolated_temp).reset_index()
			# Save metrics for each station to a CSV
			metrics_per_station_interpolated.to_csv('metrics_per_station_interpolated_' + grid + '_' + selected_variable + '.csv', index=False)
			print('metrics_per_station_interpolated_' + grid + '_' + selected_variable + '.csv has been saved')
			# Units for each metric
			units = {
				'Mean Bias': '°C',
				'P90 Bias': '°C',
				'P10 Bias': '°C',
				'Mean Absolute Error': '°C',
				'RMSE': '°C',
				'Correlation': 'Dimensionless',
				'Variance Bias': '°C²',
				'Std Bias': '°C',
				'Wasserstein Distance': '',
				'KS test stat': '',
				'KS test p': '',
				'Skew Bias': 'Dimensionless',
				'Kurtosis Bias': 'Dimensionless'
			}

		elif selected_variable == 'wind_speed':
			metrics_per_station_interpolated = stations_data.groupby('station_id').apply(calculate_metrics_interpolated_wspeed).reset_index()
			print(metrics_per_station_interpolated)
			
			# Save metrics for each station to a CSV
			metrics_per_station_interpolated.to_csv('metrics_per_station_interpolated_' + grid + '_' + selected_variable + '.csv', index=False)
			
			print('metrics_per_station_interpolated_' + grid + '_' + selected_variable + '.csv has been saved')    
			units = {
				'Mean Bias': 'm/s',
				'P95 Bias': 'm/s',
				'P90 Bias': 'm/s',
				'P10 Bias': 'm/s',
				'Mean Absolute Error': 'm/s',
				'RMSE': 'm/s',
				'Correlation': 'Dimensionless',
				'Variance Bias': 'm²/s²',
				'Std Bias': 'm/s',
				'Wasserstein Distance': '',
				'KS test stat': '',
				'KS test p': '',
				'Skew Bias': 'Dimensionless',
				'Kurtosis Bias': 'Dimensionless'
			}

		elif selected_variable == 'radiation':
			metrics_per_station_interpolated = stations_data.groupby('station_id').apply(calculate_metrics_interpolated_wspeed).reset_index()
			# Save metrics for each station to a CSV
			metrics_per_station_interpolated.to_csv('metrics_per_station_interpolated_' + grid + '_' + selected_variable + '.csv', index=False)
			print('metrics_per_station_interpolated_' + grid + '_' + selected_variable + '.csv has been saved')    
			units = {
				'Mean Bias': 'J/m²',
				'P95 Bias': 'J/m²',
				'P90 Bias': 'J/m²',
				'P10 Bias': 'J/m²',
				'Mean Absolute Error': 'J/m²',
				'RMSE': 'J/m²',
				'Correlation': 'Dimensionless',
				'Variance Bias': 'J²/m⁴',
				'Std Bias': 'J/m²',
				'Wasserstein Distance': '',
				'KS test stat': '',
				'KS test p': '',
				'Skew Bias': 'Dimensionless',
				'Kurtosis Bias': 'Dimensionless'
			}

		elif selected_variable == 'precipitation':
			metrics_per_station_interpolated = stations_data.groupby('station_id').apply(calculate_metrics_interpolated_precip).reset_index()
			
			# Save metrics for each station to a CSV
				
			units = {
				'Mean Bias': 'mm',
				'P99 Bias': 'mm',
				'P95 Bias': 'mm',
				'P90 Bias': 'mm',
				'P10 Bias': 'mm',
				'Mean Absolute Error': 'mm',
				'Correlation': 'Dimensionless',
				'Variance Bias': 'mm²',
				'Number of wet days Bias': 'days',
				'SEEPS Skill Score': 'Dimensionless',
				'Mean relative Bias': '%',
				'Std relative Bias': '%',
				'Variance relative Bias': '%',
				'P99 relative Bias': '%',
				'P95 relative Bias': '%',
				'P90 relative Bias': '%',
				'P10 relative Bias': '%',
				'Multiplicative Bias': 'Dimensionless',
				'Std Bias': 'mm',
				'Wasserstein Distance': '',
				'KS test stat': '',
				'KS test p': '',
				'Skew Bias': 'Dimensionless',
				'Kurtosis Bias': 'Dimensionless'
			}
			
			# Calculate the number of days with observed precipitation greater than 0.25 mm
			stations_data_accu_ini['days_with_precipitation'] = stations_data_accu_ini['precipitation'] > 0.25
			stations_data_accu_ini['days_with_precipitation_interpolated'] = stations_data_accu_ini['interpolated_grid_value'] > 0.25

			# Group data by station and calculate accumulated precipitation for both observed and interpolated data
			accumulated_precipitation = stations_data_accu_ini.groupby('station_id').agg({
				'precipitation': 'sum',
				'interpolated_grid_value': 'sum',
				'days_with_precipitation': 'sum',
				'days_with_precipitation_interpolated': 'sum'
			}).reset_index()

			# Calculate differences and metrics based on accumulated precipitation
			accumulated_precipitation['difference_days'] = accumulated_precipitation['days_with_precipitation_interpolated'] - accumulated_precipitation['days_with_precipitation']

			def calculate_metrics_accumulated(data):
				R0025_bias = data['difference_days'].mean()
				return pd.Series({
					'Number of wet days Bias': R0025_bias
				})

			metrics_per_station_accumulated = accumulated_precipitation.groupby('station_id').apply(calculate_metrics_accumulated).reset_index()
			
			# Save metrics for each station to a CSV
			metrics_per_station_interpolated['Number of wet days Bias'] = metrics_per_station_accumulated['Number of wet days Bias'] / (stations_data['date'].nunique() / (365 * len(selected_months) / 12))

			metrics_per_station_interpolated.to_csv('metrics_per_station_interpolated_' + grid + '_' + selected_variable + '.csv', index=False)
			print('metrics_per_station_interpolated_' + grid + '_' + selected_variable + '.csv has been saved')

		elif selected_variable == 'humidity':
			print(stations_data.groupby('station_id').apply(calculate_metrics_interpolated_humidity))
			metrics_per_station_interpolated = stations_data.groupby('station_id').apply(calculate_metrics_interpolated_humidity).reset_index()
			# Save metrics for each station to a CSV
			metrics_per_station_interpolated.to_csv('metrics_per_station_interpolated_' + grid + '_' + selected_variable + '.csv', index=False)
			print('metrics_per_station_interpolated_' + grid + '_' + selected_variable + '.csv has been saved')
			units = {
				'Mean Bias': '%',
				'P95 Bias': '%',
				'P90 Bias': '%',
				'P10 Bias': '%',
				'Mean Absolute Error': '%',
				'RMSE': '%',
				'Correlation': 'Dimensionless',
				'Variance Bias': '%²',
				'Std Bias': '%',
				'Wasserstein Distance': '',
				'KS test stat': '',
				'KS test p': '',
				'Skew Bias': 'Dimensionless',
				'Kurtosis Bias': 'Dimensionless'
			}

		else:
			print(f'Error: variable {selected_variable} not contemplated for metric calculation.')
			continue

		# Calculate the annual cycle of the grid

		dfac = stations_data.dropna()
		dfac['month'] = dfac['date'].dt.month
		dfac['interpolated_grid_value'] = dfac['interpolated_grid_value'].astype(float)
		# Group by month and calculate the annual cycle averaging over all observational stations
		if selected_variable != 'precipitation':
			# Step 1: Calculate the monthly mean per station
			station_monthly = dfac.groupby(['station_id', 'month']).agg({
				'interpolated_grid_value': 'mean'
			}).reset_index()
			
			# Step 2: Calculate the global monthly mean across stations
			monthly_avg = station_monthly.groupby('month').agg({
				'interpolated_grid_value': 'mean'
			}).reset_index()
		else:
			# Step 1: Calculate the monthly sum per station
			station_monthly = dfac.groupby(['station_id', 'month']).agg({
				'interpolated_grid_value': 'sum'
			}).reset_index()
			
			# Step 2: Divide by the number of years in the analysis period
			num_years = stations_data['date'].nunique() / (365 * len(selected_months) / 12)  
			station_monthly['interpolated_grid_value'] /= num_years
			
			# Step 3: Calculate the global monthly mean across all stations
			monthly_avg = station_monthly.groupby('month').agg({
				'interpolated_grid_value': 'mean'
			}).reset_index()
		# Export to CSV
		monthly_avg.to_csv(selected_variable + '_' + grid + '_annual_cycle_comparison.csv', index=False)

		del stations_data
		del stations_data_accu_ini
		del dfac
	
	# Calculate the annual cycle of observations

	stations_data_0['month'] = stations_data_0['date'].dt.month
	stations_data_0[selected_variable] = stations_data_0[selected_variable].astype(float)
	# Group by month and calculate the annual cycle averaging over all observational stations
	if selected_variable != 'precipitation':
		# Step 1: Calculate monthly mean per station
		station_monthly_obs = stations_data_0.groupby(['station_id', 'month']).agg({
			selected_variable: 'mean'
		}).reset_index()
			
		# Step 2: Calculate global monthly mean across stations
		monthly_avg_obs = station_monthly_obs.groupby('month').agg({
			selected_variable: 'mean'
		}).reset_index()
	else:
		# Step 1: Calculate monthly sum per station
		station_monthly_obs = stations_data_0.groupby(['station_id', 'month']).agg({
			selected_variable: 'sum'
		}).reset_index()
		
		# Step 2: Divide by the number of years in the analysis period
		num_years = stations_data_0['date'].nunique() / (365 * len(selected_months) / 12)
		station_monthly_obs[selected_variable] /= num_years
		
		# Step 3: Calculate global monthly mean across all stations
		monthly_avg_obs = station_monthly_obs.groupby('month').agg({
			selected_variable: 'mean'
		}).reset_index()
	# Export to CSV
	monthly_avg_obs.to_csv(selected_variable + '_annual_cycle_obs.csv', index=False)


	# Dictionary to store metric data
	metrics_data_dict = {}
	# Dictionary to store annual cycle data
	annual_cycle_dict = {}

	# Load metrics for each grid from CSVs
	for grid in selected_grids:  
		file_name = f'metrics_per_station_interpolated_{grid}_{selected_variable}.csv'
		file_name_annual_cycle = selected_variable + '_' + grid + '_annual_cycle_comparison.csv'
		
		try:
			metrics_data = pd.read_csv(file_name)
			metrics_data_dict[grid] = metrics_data
			
			annual_cycle_data = pd.read_csv(file_name_annual_cycle)
			annual_cycle_dict[grid] = annual_cycle_data
			
			# Generate maps for each metric
			# Load station data CSV
			stations_df = pd.read_csv("stations_completeness.csv")
			# Merge the two DataFrames by station_id
			merged_df = pd.merge(stations_df, metrics_data, on="station_id", how="inner")
			metrics_to_plot = [col for col in metrics_data.columns if col != "station_id"]
			# Generate a scatterplot for each metric
			for metric in metrics_to_plot:
				# Set projection and create figure
				fig, ax = plt.subplots(
					figsize=(10, 8),
					subplot_kw={'projection': ccrs.PlateCarree()}
				)
				# Add high-resolution coastlines
				ax.coastlines(resolution='10m', color='black', linewidth=1)
				# Add borders
				ax.add_feature(cfeature.BORDERS, linestyle='--')
				# Plot station data
				scatter = ax.scatter(
					merged_df['longitude'], merged_df['latitude'],
					c=merged_df[metric], cmap='viridis', s=100, edgecolor='k', transform=ccrs.PlateCarree()
				)
				# Configure colorbar
				if metric not in ['Wasserstein Distance', 'KS test stat', 'KS test p']:
					colorbar = plt.colorbar(scatter, ax=ax, label=f'{metric} ({units.get(metric, "")})')
				else:
					colorbar = plt.colorbar(scatter, ax=ax, label=f'{metric} {units.get(metric, "")}')
				# Set color limits if metric is SEEPS Skill Score
				if metric == "SEEPS Skill Score":
					scatter.set_clim(0, 1)
				# Titles and labels
				plt.title(f"{metric} comparison between {grid} and stations")
				plt.xlabel("Longitude")
				plt.ylabel("Latitude")
				# Save and close figure
				plt.savefig(f"map_{metric.replace(' ', '_').lower()}_{grid}_{selected_variable}.png")
				plt.close()
			
		except FileNotFoundError:
			print(f'Warning: No metrics file found for {grid} and {selected_variable}')

	# Load the annual cycle of observations from CSV
	file_name_annual_cycle_obs = selected_variable + '_annual_cycle_obs.csv'
	annual_cycle_data_obs = pd.read_csv(file_name_annual_cycle_obs)

	# Verify loaded data
	for grid, metrics_data in metrics_data_dict.items():
		print(f'Loaded metrics data for {grid} and {selected_variable}')

	# Create a list of dataframes for seaborn
	dfs = list(metrics_data_dict.values())

	# Concatenate dataframes
	metrics_concat = pd.concat(dfs, keys=metrics_data_dict.keys(), names=['Grid'])

	# Reset index completely to ensure it's simple
	metrics_concat = metrics_concat.reset_index()

	# Remove 'level_1' column if it exists
	if 'level_1' in metrics_concat.columns:
		metrics_concat = metrics_concat.drop('level_1', axis=1)
		
	# Automatically get the list of available metrics
	metrics_to_plot = metrics_concat.drop(['Grid', 'station_id'], axis=1).columns.tolist()

	# Generate boxplots for each metric
	for metric in metrics_to_plot:
		plt.figure(figsize=(10, 6))
		sns.boxplot(data=metrics_concat, x='Grid', y=metric, hue=None, orient='v', dodge=False)
		plt.title(f'Comparison of {metric} for {selected_variable}')
		if metric not in ['Wasserstein Distance', 'KS test stat', 'KS test p']:
			plt.ylabel(f'{metric} ({units.get(metric, "")})')
		else:
			plt.ylabel(f'{metric} {units.get(metric, "")}')
		plt.xlabel('Grid')
		plt.xticks(rotation=45)
		plt.tight_layout()
		plt.savefig(f'{selected_variable}_{metric}_grids_comparison.png')
		print(f'{selected_variable}_{metric}_grids_comparison.png has been saved')
		plt.close()
		
	# Plot the annual cycle
	plt.figure(figsize=(12, 6))
	plt.plot(annual_cycle_dict[grid]['month'], annual_cycle_data_obs[selected_variable], label='observations', marker='o', color='black')
	for grid in selected_grids:
		plt.plot(annual_cycle_dict[grid]['month'], annual_cycle_dict[grid]['interpolated_grid_value'], label=grid, marker='o')
	# Configure plot
	plt.xlabel('Month')
	if selected_variable == 'precipitation':
		plt.ylabel(selected_variable + ' (mm)')
	elif selected_variable == 'wind_speed':
		plt.ylabel(selected_variable + ' (m/s)')
	elif selected_variable == 'humidity':
		plt.ylabel(selected_variable + ' (%)')
	elif selected_variable == 'radiation':
		plt.ylabel(selected_variable + ' (J/m²)')
	else:
		plt.ylabel(selected_variable + ' (°C)')
	plt.title('Average Annual Cycle')
	# Abbreviated month labels in English
	months_abbr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
	months_names = [months_abbr[month - 1] for month in selected_months]
	plt.xticks(annual_cycle_dict[grid]['month'], months_names)
	plt.legend()
	plt.savefig(f'{selected_variable}_annual_cycle_grids_comparison.png')
	plt.close()
	print(f'{selected_variable}_annual_cycle_grids_comparison.png has been saved')

	# Accumulators
	all_obs = []
	rean_data = {}

	# Iterate through all grids
	for grid in selected_grids:
		filename = f'{selected_variable}_stations_data_and_interpolated_grid_value_{grid}.csv'
		df = pd.read_csv(filename)

		if selected_variable not in df.columns:
			raise ValueError(f"Column '{selected_variable}' not found in {filename}")
		if 'interpolated_grid_value' not in df.columns:
			raise ValueError(f"Column 'interpolated_grid_value' not found in {filename}")

		# Observations (assumed to be the same in all files)
		all_obs.append(df[selected_variable])

		# Reanalysis
		rean_data[grid] = df['interpolated_grid_value'].dropna().values

	# Concatenate observations
	obs_all = pd.concat(all_obs).dropna().values

	# Helper functions
	def estimate_pdf(data, bw_method='scott'):
		kde = gaussian_kde(data, bw_method=bw_method)
		x = np.linspace(np.min(data), np.max(data), 500)
		y = kde(x)
		return x, y

	# If precipitation, filter zeros or very small values
	if selected_variable == 'precipitation':
		obs_all = obs_all[obs_all > 0.1]  # filter observations
		
	# Estimate PDF of observations
	x_obs, pdf_obs = estimate_pdf(obs_all)

	# Create figure
	plt.figure(figsize=(10, 6))

	# Plot PDF of observations
	plt.plot(x_obs, pdf_obs, label='Observations', color='black', linewidth=2)

	# Plot PDFs of reanalysis
	for grid, data in rean_data.items():
		if selected_variable == 'precipitation':
			data = data[data > 0.1]  # filter grids as well
		x_r, pdf_r = estimate_pdf(data)
		plt.plot(x_r, pdf_r, label=grid)

	# Labels
	if selected_variable == 'precipitation':
		plt.xlabel('Precipitation (mm)')
		plt.xscale('log')
		plt.ylim(0, None)
	elif selected_variable == 'wind_speed':
		plt.xlabel('Wind speed (m/s)')
	elif selected_variable == 'humidity':
		plt.xlabel('Humidity (%)')
	elif selected_variable == 'radiation':
		plt.xlabel('Radiation (J/m²)')
	else:
		plt.xlabel(selected_variable)

	plt.ylabel('PDF')
	plt.title('PDF comparison: Observations vs datasets')
	plt.legend()
	plt.grid(True)
	plt.tight_layout()

	# Save figure
	plt.savefig(f'{selected_variable}_pdf_comparison_stations_vs_datasets.png', dpi=300)
	plt.close()

	
def on_generate_button_click():
	selected_variables = [combo_variable.get()]
	selected_grids = listbox_grids.curselection()
	selected_grids = [grids[i] for i in selected_grids]
	interpolation_method = interpolation_var.get()
	start_year = entry_start_year.get()
	end_year = entry_end_year.get()
	period_type = period_var.get()

	# Validate year inputs
	if not start_year.isdigit() or not end_year.isdigit():
		messagebox.showerror("Error", "Please enter valid years.")
		return
	if int(start_year) > int(end_year):
		messagebox.showerror("Error", "Start year must be less than or equal to end year.")
		return
	if period_type == "Anual":
		selected_months = list(range(1, 13))  # All months
	else:
		selected_months = [i + 1 for i, var in enumerate(month_vars) if var.get() == 1]  # Selected months
		
	generate_metrics_and_plots(selected_grids, selected_variables[0], int(start_year), int(end_year), selected_months, interpolation_method)
	
# Variables and grid list
variables = ['temperature', 'maximum_temperature', 'minimum_temperature', 'precipitation', 'wind_speed', 'humidity', 'radiation']
grids = ['HCLIM_IBERIAxm_40', 'HCLIM_IBERIAxxm_40', 'HCLIM_IBERIAxxs_10', 'WRF', 'ISIMIP-CHELSA', 'CHIRTS', 'CHIRPS', 'ERA5', 'ERA5-Land', 'COSMO-REA6', 'CERRA', 'CERRA_LAND', 'EOBS', 'EOBS_HR', 'EOBS_LR', 'CLARA-A3']
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']


# Create the GUI
root = tk.Tk()
root.title('Grid Evaluator')

# Labels and ComboBox to select variable
label_variable = ttk.Label(root, text='Select Variable:')
label_variable.pack(pady=10)
combo_variable = ttk.Combobox(root, values=variables)
combo_variable.pack()

# Label and Listbox to select grids
label_grids = ttk.Label(root, text='Select Grids:')
label_grids.pack(pady=10)
listbox_grids = tk.Listbox(root, selectmode=tk.MULTIPLE, exportselection=0)
for grid in grids:
    listbox_grids.insert(tk.END, grid)
listbox_grids.pack()

# Select interpolation method
label_interpolation = ttk.Label(root, text='Select Interpolation Method:')
label_interpolation.pack(pady=10)

interpolation_var = tk.StringVar(value="nearest")  # Default value

frame_interpolation = ttk.Frame(root)
frame_interpolation.pack(pady=10)

radiobutton_nearest = ttk.Radiobutton(frame_interpolation, text="Nearest Neighbor", variable=interpolation_var, value="nearest")
radiobutton_nearest.grid(row=0, column=0, padx=5)

radiobutton_bilinear = ttk.Radiobutton(frame_interpolation, text="Bilinear", variable=interpolation_var, value="linear")
radiobutton_bilinear.grid(row=0, column=1, padx=5)

radiobutton_15kmrelax = ttk.Radiobutton(frame_interpolation, text="15 km relaxation", variable=interpolation_var, value="15km relaxation")
radiobutton_15kmrelax.grid(row=0, column=2, padx=5)

# Labels and entries to select period
label_period = ttk.Label(root, text='Select Period:')
label_period.pack(pady=10)

frame_period = ttk.Frame(root)
frame_period.pack(pady=10)

label_start_year = ttk.Label(frame_period, text='Start Year:')
label_start_year.grid(row=0, column=0, padx=5)

entry_start_year = ttk.Entry(frame_period, width=10)
entry_start_year.grid(row=0, column=1, padx=5)
entry_start_year.insert(0, '1991')  # Default start year

label_end_year = ttk.Label(frame_period, text='End Year:')
label_end_year.grid(row=0, column=2, padx=5)

entry_end_year = ttk.Entry(frame_period, width=10)
entry_end_year.grid(row=0, column=3, padx=5)
entry_end_year.insert(0, '2020')  # Default end year

# Period type selection
period_var = tk.StringVar(value="Anual")

frame_seasonal = ttk.Frame(root)
frame_seasonal.pack(pady=10)

radiobutton_annual = ttk.Radiobutton(frame_seasonal, text="Annual", variable=period_var, value="Anual")
radiobutton_annual.grid(row=0, column=0, padx=5)

radiobutton_monthly = ttk.Radiobutton(frame_seasonal, text="Custom Months", variable=period_var, value="Mensual")
radiobutton_monthly.grid(row=0, column=1, padx=5)

# Checkboxes to select months
frame_months = ttk.Frame(root)
frame_months.pack(pady=10)

month_vars = []
for i, month in enumerate(months):
    var = tk.IntVar(value=0)
    month_vars.append(var)
    checkbutton = ttk.Checkbutton(frame_months, text=month, variable=var)
    checkbutton.grid(row=i // 4, column=i % 4, sticky="w", padx=5)    

# Button to generate metrics and plots
generate_button = ttk.Button(root, text='Generate Metrics & Plots', command=on_generate_button_click)
generate_button.pack(pady=20)

# Start the GUI
root.mainloop()
