import pvlib
from pvlib.location import Location
import pandas as pd
import numpy as np


locations = pd.read_csv("valsequillo_fpv_CLIM.csv", encoding='latin1')


start_year = 2014
end_year = 2023
altitude = 2059
raddatabase = 'PVGIS-ERA5'
tz = "America/Mexico_City"



hourly_data = {}
daily_data = {}



for index, row in locations.iterrows():
    name = row['unit']
    latitude = row['lat']
    longitude = row['long']

    print(f"Processing {name}...")

   
    location = Location(latitude, longitude, tz=tz, altitude=altitude, name=name)
    times = pd.date_range(start=f'{start_year}-01-01 00:00', end=f'{end_year}-12-31 23:00',
                          freq="H", tz=location.tz)
    times_df = pd.DataFrame(index=times)
    solpos = location.get_solarposition(times = times)
    daily_solar_zenith = solpos["zenith"].resample("D").mean()

   


    poa, inputs, metadata = pvlib.iotools.get_pvgis_hourly(
        latitude=latitude,
        longitude=longitude,
        start=start_year,
        end=end_year,
        raddatabase=raddatabase,
        components=True,
        outputformat='json',
        usehorizon=True,
        userhorizon=None,
        pvcalculation=True,
        peakpower=1221,
        pvtechchoice='crystSi',
        mountingplace='free',
        trackingtype=0,
        url='https://re.jrc.ec.europa.eu/api/v5_3/',
        map_variables=True,
        timeout=120
    )
    poa = poa.resample("h").mean()
    poa.index = poa.index.tz_convert(location.tz)
    extra=pvlib.irradiance.get_extra_radiation(times)
    extra=extra.reindex(poa.index)
    

    
    poa['poa_global']=poa['poa_direct']+poa['poa_sky_diffuse']+poa['poa_ground_diffuse']
    clim = pd.DataFrame(index=poa.index)
    clim['POA_global']=poa['poa_global']

    clim["Temp [°C]"] = poa["temp_air"]
    clim["Wind_2m [m/s]"] = poa['wind_speed'] * np.log(2 / 0.001) / np.log(10 / 0.001)

    clim['hourly_SCI_kt'] = pvlib.irradiance.clearness_index(poa['poa_global'], solpos["zenith"], extra, max_clearness_index=1.0)

    hourly_data[name] = clim
    
    
    

    daily = pd.DataFrame(index=clim.resample("D").mean().index)
    daylight_mask = poa['poa_global'] > 0  

    
    sci_daylight = hourly_data[name]['hourly_SCI_kt'][daylight_mask]
    
    
    daily['SCI_kt'] = sci_daylight.resample('D').sum() / daylight_mask.resample('D').sum()
    daily["Temp [°C]"] = clim["Temp [°C]"].resample("D").mean()
    daily["Temp [°F]"] = (daily["Temp [°C]"] * 9/5) + 32
    
    daily["Radiation [W/m²]"] = poa['poa_global'].resample("D").sum() / ((poa['poa_global'] > 0).resample("D").sum())
    daily["Radiation [MJ/m²/day]"] = daily["Radiation [W/m²]"] * 24 * 3600 / 1e6
    daily_data[name] = daily


hourly_temp_df = pd.DataFrame()
hourly_poaglobal_df = pd.DataFrame()
hourly_wind_df = pd.DataFrame()
hourly_sci_df = pd.DataFrame()  
daily_temp_df = pd.DataFrame()
daily_radiation_df = pd.DataFrame()
daily_sci_df = pd.DataFrame()  
daily_sci_df = pd.DataFrame()  


# Assemble combined data
for name in locations['unit']:
    # Hourly data
    temp_hourly = hourly_data[name]["Temp [°C]"].copy()
    temp_hourly.index = temp_hourly.index.tz_localize(None)
    hourly_temp_df[name] = temp_hourly

    wind_hourly = hourly_data[name]["Wind_2m [m/s]"].copy()
    wind_hourly.index = wind_hourly.index.tz_localize(None)
    hourly_wind_df[name] = wind_hourly
    
    sci_hourly = hourly_data[name]["hourly_SCI_kt"].copy()
    sci_hourly.index = sci_hourly.index.tz_localize(None)
    hourly_sci_df[name] = sci_hourly
    
    poaglobal_hourly = hourly_data[name]['POA_global']
    poaglobal_hourly.index = poaglobal_hourly.index.tz_localize(None)
    hourly_poaglobal_df[name] = poaglobal_hourly
    
    

    temp_daily = daily_data[name]["Temp [°C]"].copy()
    temp_daily.index = temp_daily.index.tz_localize(None)
    daily_temp_df[name] = temp_daily

    radiation_daily = daily_data[name]["Radiation [MJ/m²/day]"].copy()
    radiation_daily.index = radiation_daily.index.tz_localize(None)
    daily_radiation_df[name] = radiation_daily


    sci_daily = daily_data[name]["SCI_kt"].copy()
    sci_daily.index = sci_daily.index.tz_localize(None)
    daily_sci_df[name] = sci_daily



# =============================================================================
# Visualization - Integrated 3x1 Figure for Vlasequillo
# =============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# Extract the column for 'Valsequillo' only
valsequillo_temp = hourly_temp_df['Valsequillo']
valsequillo_wind = hourly_wind_df['Valsequillo']
valsequillo_sci_daily = daily_sci_df['Valsequillo']

# Set styles
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.5)

# Create layout
fig = plt.figure(figsize=(20, 16), constrained_layout=True)
gs = GridSpec(3, 1, height_ratios=[1, 1, 1.2], figure=fig)

# --- 1. Temperature Heatmap ---
ax1 = fig.add_subplot(gs[0])
temp_hourly = valsequillo_temp.copy()
temp_hourly.index = pd.to_datetime(temp_hourly.index)
temp_heatmap_data = temp_hourly.groupby(
    [temp_hourly.index.date, temp_hourly.index.hour]
).mean().unstack()
temp_heatmap_data.index = pd.to_datetime(temp_heatmap_data.index)

sns.heatmap(temp_heatmap_data.T, cmap='coolwarm', 
            cbar_kws={'label': 'Temperature (°C)'}, 
            xticklabels=60, ax=ax1)

ax1.set_title(f"A) Hourly Temperature Heatmap – Valsequillo ({start_year}–{end_year})", fontweight='bold', fontsize=18, loc='left', pad=15)
ax1.set_xlabel("")
ax1.set_ylabel("Hour of Day", fontsize=15)
ax1.tick_params(axis='y', labelsize=13)

# Format x-axis
formatted_index = temp_heatmap_data.index.to_series().dt.to_period("M").astype(str).tolist()
tick_locs = np.linspace(0, len(formatted_index)-1, 12, dtype=int)
tick_labels = [formatted_index[i] for i in tick_locs]
ax1.set_xticks(tick_locs)
ax1.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=13)

# --- 2. Wind Speed Heatmap ---
ax2 = fig.add_subplot(gs[1])
wind_hourly = valsequillo_wind.copy()
wind_hourly.index = pd.to_datetime(wind_hourly.index)
wind_heatmap_data = wind_hourly.groupby(
    [wind_hourly.index.date, wind_hourly.index.hour]
).mean().unstack()
wind_heatmap_data.index = pd.to_datetime(wind_heatmap_data.index)

sns.heatmap(wind_heatmap_data.T, cmap='YlGnBu', 
            cbar_kws={'label': 'Wind Speed (m/s)'}, 
            xticklabels=60, ax=ax2)

ax2.set_title(f"B) Hourly Wind Speed Heatmap – Valsequillo ({start_year}–{end_year})", fontweight='bold', fontsize=18, loc='left', pad=15)
ax2.set_xlabel("")
ax2.set_ylabel("Hour of Day", fontsize=15)
ax2.tick_params(axis='y', labelsize=13)

# Format x-axis
ax2.set_xticks(tick_locs)
ax2.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=13)

# --- 3. SCI Daily Time Series ---
ax3 = fig.add_subplot(gs[2])
sci_daily = valsequillo_sci_daily.copy()
sci_daily.index = pd.to_datetime(sci_daily.index)

# Plot scatter and monthly line
ax3.scatter(sci_daily.index, sci_daily, color='steelblue', s=10, alpha=0.4, label='Daily SCI')
monthly_sci = sci_daily.resample('M').mean()
ax3.plot(monthly_sci.index, monthly_sci, color='darkorange', linewidth=2, label='Monthly Average')

# Classification lines
ax3.axhline(0.7, color='green', linestyle='--', linewidth=1.8)
ax3.axhline(0.6, color='orange', linestyle='--', linewidth=1.8)
ax3.axhline(0.3, color='red', linestyle='--', linewidth=1.8)

# Text annotations
label_x = sci_daily.index[-1] + pd.Timedelta(days=90)
ax3.text(label_x, 0.70, 'Clear Sky (KT ≥ 0.7)', color='green', fontsize=14)
ax3.text(label_x, 0.60, 'Partly Cloudy (0.3 ≤ KT < 0.7)', color='orange', fontsize=14)
ax3.text(label_x, 0.30, 'Overcast (KT < 0.3)', color='red', fontsize=14)

# Format
ax3.set_title(f"C) Daily Average Sky Clearness Index (SCI) – Valsequillo ({start_year}–{end_year})", fontweight='bold', fontsize=18, loc='left', pad=15)
ax3.set_xlabel("Year", fontsize=15)
ax3.set_ylabel("Sky Clearness Index (KT)", fontsize=15)
ax3.set_ylim(0, 1.05)
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.legend(fontsize=13, loc='upper right')

# Time axis formatting
ax3.xaxis.set_major_locator(mdates.YearLocator())
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax3.xaxis.set_minor_locator(mdates.MonthLocator())
ax3.tick_params(axis='both', which='major', labelsize=13)

plt.subplots_adjust(hspace=0.35)
plt.show()
# =============================================================================
# Classify daily SCI values by cloud condition
# =============================================================================
# Classify SCI condition
def classify_sci(sci_value):
    if sci_value >= 0.7:
        return 'Clear Sky'
    elif sci_value >= 0.3:
        return 'Partly Cloudy'
    elif sci_value < 0.3 and sci_value > 0:  # Fixed: changed && to and
        return 'Overcast'
    elif sci_value == 0:
        return 'Night'

# DataFrame to store classification results
classification_summary = pd.DataFrame()

# Loop through each unit
for unit in hourly_sci_df.columns:
    sci_series = hourly_sci_df[unit].copy()
    sci_series.index = pd.to_datetime(sci_series.index)
    
    # Apply classification
    classified = sci_series.apply(classify_sci)
    
    # Group by year and condition
    summary = classified.groupby([classified.index.year, classified]).size().unstack(fill_value=0)
    summary['Unit'] = unit
    summary['Year'] = summary.index
    
    classification_summary = pd.concat([classification_summary, summary], axis=0)


classification_summary = classification_summary.reset_index(drop=True)
classification_summary = classification_summary[['Unit', 'Year', 'Clear Sky', 'Partly Cloudy', 'Overcast', 'Night']]
# =============================================================================
# 
# -----------------------------------------------------------------------------
# Distribution Plots for Temperature and Wind Speed – Valsequillo (Hourly Data)
# -----------------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt

valsequillo_temp = hourly_temp_df['Valsequillo'].dropna()
valsequillo_wind = hourly_wind_df['Valsequillo'].dropna()

sns.set_style("whitegrid")
sns.set_context("talk", font_scale=1.2)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

sns.histplot(valsequillo_temp, bins=50, kde=True, color='salmon', ax=axes[0])
axes[0].set_title("A) Temperature Distribution (°C) – Valsequillo (2014–2023)", loc='left', fontsize=16, fontweight='bold')
axes[0].set_xlabel("Temperature (°C)")
axes[0].set_ylabel("Frequency")

# --- Wind Speed Distribution ---
sns.histplot(valsequillo_wind, bins=50, kde=True, color='steelblue', ax=axes[1])
axes[1].set_title("B) Wind Speed Distribution (m/s) – Valsequillo (2014–2023)", loc='left', fontsize=16, fontweight='bold')
axes[1].set_xlabel("Wind Speed (m/s)")
axes[1].set_ylabel("Frequency")
plt.tight_layout()
plt.show()

# =============================================================================

# Save to Excel
with pd.ExcelWriter("Valsequillo_CLIM_results2.xlsx") as writer:
    hourly_temp_df.to_excel(writer, sheet_name="Hourly_Temp_C")
    hourly_wind_df.to_excel(writer, sheet_name="Hourly_Wind2m_mps")
    hourly_sci_df.to_excel(writer, sheet_name="Hourly_SCI_kt")
    hourly_poaglobal_df.to_excel(writer, sheet_name="Hourly_POA_Global_W_m2")
    # hourly_csghi_df.to_excel(writer, sheet_name="Hourly_Clear_Sky_GHI_W_m2")
    daily_temp_df.to_excel(writer, sheet_name="Daily_Temp_C")
    daily_radiation_df.to_excel(writer, sheet_name="Daily_Radiation_MJ")
    daily_sci_df.to_excel(writer, sheet_name="Daily_SCI")
    classification_summary.to_excel(writer, sheet_name="SCI_Classification", index=False)


