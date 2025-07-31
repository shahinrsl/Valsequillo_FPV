import pvlib
import pvlib.irradiance 
from pvlib.location import Location
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy_financial as npf

# Load locations data
locations = pd.read_csv("valsequillo_fpv_TEA.csv", encoding='latin1')
PVsystemconfig = 'LPV' # LPV or FPV

# Initialize new columns for results
base_year = 2022
altitude = 2059
optimum_azimuth_angle = -16
surface_azimuth = optimum_azimuth_angle + 180
poa_surface_azimuth = optimum_azimuth_angle
raddatabase = 'PVGIS-ERA5'
tz = "America/Mexico_City"
Pmax_STC = 550
inverter_pdc0 = 600
operation_period = 25
gamma_pdc = -0.0034

cel_rate = 1 # 1 cel/MWh
cel_price = 20.57 #USD/CEL

if PVsystemconfig == 'FPV':
    land_price= 1e-10 # USD/ha to avoid problems in random values generation 
    module=315.7 # USD/kWdc
    inverter=55.0 # USD/kWdc
    bos_hardware=469.98 # USD/kWdc
    installation=139.5 # USD/kWdc
    soft_costs=255.1 # USD/kWdc
    surface_tilt = 10
    opex= 19 # USD/kWdc
    opex_upper=34.86
    opex_lower=9.05
    
elif PVsystemconfig == 'LPV':
    land_price=30000 #USD/ha
    module=315.7 # USD/kWdc
    inverter=55.0 # USD/kWdc
    bos_hardware=294.7 # USD/kWdc
    installation=129.5 # USD/kWdc
    soft_costs=255.1 # USD/kWdc
    surface_tilt = 21 # degree
    opex=7.56 # USD/kWdc
    opex_upper=13.87
    opex_lower=3.6
    
capex_t0=module+inverter+bos_hardware+installation+soft_costs # USD/kWdc
inverter_cost = inverter # USD/Wdc

discount_rate =6.0/100 # same value for Aguascalientes utility scale PV
discount_rate_upper=7.6/100
discount_rate_lower=4.6/100

degradation_rate=0.58/100 # 0.6% degradation rate per year 

# =============================================================================
ac_results =pd.DataFrame()

years = range(base_year+1, base_year+operation_period+1)
ac_results = []

for year in years:
    # Create a date range for one year (excluding Feb 29)
    start = f'{year}-01-01 00:00'
    end = f'{year}-12-31 23:00'
    dr = pd.date_range(start, end, freq='H')
    
    # Remove Feb 29 if it exists
    dr = dr[~((dr.month == 2) & (dr.day == 29))]
    
    ac_results.append(pd.DataFrame({'datetime': dr}))

# Combine all years
ac_results = pd.concat(ac_results, ignore_index=True)


# ac_results['operation_year'] = ac_results['datetime'].dt.year

# =============================================================================



for index, row in locations.iterrows():
    name = row['unit']  
    latitude = row['lat']
    longitude = row['long']
    num_panels = row ['no_panels']

    # Location and solar position
    location = Location(latitude,
                        longitude, 
                        tz=tz,
                        altitude=altitude,
                        name= name)

    # Time period of the study
    times = pd.date_range(start='2022-01-01 00:00', end='2022-12-31 23:00',
                          freq="h", tz=location.tz)
    times_df = pd.DataFrame(index=times)
    # Retreiving solar positions for the study location and time period 
    solpos = location.get_solarposition(times = times)
    daily_solar_zenith = solpos["zenith"].resample("D").mean()

    # aoi and iam calculations 
    aoi = pvlib.irradiance.aoi(
        solpos.apparent_zenith, solpos.azimuth, surface_tilt, surface_azimuth)
    iam = pvlib.iam.martin_ruiz(aoi, a_r=0.155)

    poa, inputs, metadata = pvlib.iotools.get_pvgis_hourly(latitude = latitude,
                                   longitude = longitude,
                                   start= base_year - 1,
                                   end= base_year + 1,
                                   raddatabase = raddatabase,
                                   components = True,
                                   surface_azimuth = poa_surface_azimuth,
                                   surface_tilt= surface_tilt,
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
                                   timeout=30)
    
    poa = poa.resample("h").mean() 
    poa.index = poa.index.tz_convert(location.tz)
    poa = poa.loc[poa.index.intersection(times_df.index)]
    poa['poa_diffuse'] = poa['poa_sky_diffuse'] + poa['poa_ground_diffuse']
    poa['poa_global'] = poa['poa_diffuse'] + poa['poa_direct']
    effective_irradiance = poa['poa_direct']*iam + poa['poa_diffuse']

    weather_pv = pd.DataFrame(index=poa.index)
    weather_pv["wind_2m"]=poa['wind_speed']*np.log(2/0.001)/np.log(10/0.001)
    weather_pv["temp_air"] = poa["temp_air"]
    weather_pv["ghi"] = poa["poa_global"]
    if PVsystemconfig == 'FPV':
        weather_pv["cell_temperature"] = pvlib.temperature.faiman(
            poa_global=weather_pv["ghi"],
            temp_air=weather_pv["temp_air"],
            wind_speed=weather_pv["wind_2m"],
            u0=35.3,
            u1=8.9
            )
    else:
        weather_pv = pd.DataFrame(index=poa.index)
        weather_pv["wind_speed"]=poa['wind_speed']
        weather_pv["temp_air"] = poa["temp_air"]
        weather_pv["ghi"] = poa["poa_global"]
        weather_pv["cell_temperature"] = pvlib.temperature.faiman(
            weather_pv["ghi"],
            weather_pv["temp_air"],
            weather_pv["wind_speed"],
            u0=25.0,
            u1=6.84)
            
    
    for year in range(1, operation_period + 1):
        operation_year = base_year + year
        panel_performance = (1 - degradation_rate) ** (year - 1)
        
        pv_dc = pvlib.pvsystem.pvwatts_dc(effective_irradiance,
                                          weather_pv["cell_temperature"],
                                          pdc0=Pmax_STC * panel_performance,
                                          gamma_pdc=gamma_pdc,
                                          temp_ref=25)
        pv_ac = pvlib.inverter.pvwatts(pdc=pv_dc,
                                       pdc0=inverter_pdc0,
                                       eta_inv_nom=0.96,
                                       eta_inv_ref=0.9637)
        
        total_ac_output = num_panels * pv_ac
        
        output_col = f'total_ac_output_[wh]_{name}'
        
        if output_col not in ac_results.columns:
            ac_results[output_col] = 0
            
        # Add the results for this location
        year_mask = (ac_results['datetime'].dt.year == operation_year)
        ac_results.loc[year_mask, output_col] = total_ac_output.values

cols = ['datetime'] + [col for col in ac_results.columns if col not in ['datetime']]
ac_results = ac_results[cols]


ac_results = ac_results[cols]
ac_results = ac_results.set_index('datetime')

daily_ac=ac_results.resample('D').sum()
daily_ac = daily_ac[~((daily_ac.index.month == 2) & (daily_ac.index.day == 29))]
yearly_ac=ac_results.resample('Y').sum()
yearly_ac.index=yearly_ac.index.year
# =============================================================================
# Hourly Energy Price 
import pandas as pd
import numpy as np


df = pd.read_csv("02VSE-115_PML_usd_wh_2023.csv")


df_long = df.melt(id_vars=["Day", "Hour"], var_name="Month", value_name="Price")


month_map = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
    'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
    'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}
df_long["Month"] = df_long["Month"].map(month_map)


df_long["Year"] = 2023
df_long["Hour"] = df_long["Hour"].astype(int) - 1


def safe_datetime(row):
    try:
        return pd.Timestamp(int(row["Year"]), int(row["Month"]), int(row["Day"]), int(row["Hour"]))
    except:
        return pd.NaT

df_long["datetime"] = df_long.apply(safe_datetime, axis=1)
df_long = df_long.dropna(subset=["datetime"])
hourly_prices = df_long.set_index("datetime")[["Price"]].sort_index()
full_hourly_prices = pd.DataFrame()

for year in range(base_year+1, base_year+operation_period+1):
    year_prices = hourly_prices.copy()
    year_prices.index = year_prices.index.map(lambda x: x.replace(year=year))
    full_hourly_prices = pd.concat([full_hourly_prices, year_prices])

print("Current timezone of prices:", hourly_prices.index.tz)
print("Current timezone of energy:", ac_results.index.tz)
# =============================================================================
# Techno-Economic Analysis Functions 
# Assumptions 
inverter_replacement_years = [10, 20] #inverter replacement at year 10 and 20
inverter_depreciation = 0.5
# 50% value loss for 5 years of operation of inverters replaced at year 20
def costflow(capex_t0,opex,inverter_cost, land_cost, years = operation_period, inverter_years=inverter_replacement_years):
    cost=[]
    for i in range(years+1):
        if i == 0:
            cost.append(-capex_t0-land_cost)
        elif i in inverter_replacement_years:
            cost.append(-1 * (opex + inverter_cost))
        else:
            cost.append(-opex)
            
    costflow=pd.Series(cost)
    return costflow 

def revenueflow(hourly_energy, inverter_cost, land_cost, full_hourly_prices, cel_rate, cel_price,
                years=operation_period, dep=inverter_depreciation):
    yearly_revenue = []
    hourly_revenue_list = []
    
    for i in range(years + 1):
        if i == 0:
            yearly_revenue.append(0)
            # Create empty hourly revenue for year 0
            hourly_revenue_list.append(pd.Series(0, index=hourly_energy.index[hourly_energy.index.year == base_year]))
            continue
            
        current_year = base_year + i
        year_mask = (hourly_energy.index.year == current_year)
        
        if not year_mask.any():
            yearly_revenue.append(0)
            hourly_revenue_list.append(pd.Series(0, index=hourly_energy.index[year_mask]))
            continue
            
        year_energy = hourly_energy.loc[year_mask].iloc[:, 0]
        year_prices = full_hourly_prices.loc[year_mask]['Price']

        
        
        
        hourly_revenue = (year_energy * year_prices) + (year_energy * cel_price * cel_rate / 1e6)
        hourly_revenue_list.append(hourly_revenue)
        
        
        energy_revenue = hourly_revenue.sum()
        
        
        if i == years:
            energy_revenue += inverter_cost * (1 - dep) + land_cost
            
        yearly_revenue.append(energy_revenue)
    
    
    hourly_revenue_df = pd.concat(hourly_revenue_list)
    
    return pd.Series(yearly_revenue), hourly_revenue_df

def calculate_TEA_metrics(cost, revenue, rate, years=operation_period):
    cashflow = cost + revenue
    cashflow = pd.DataFrame(cashflow, columns=['UndiscountedCashFlows'])
    
    
    cashflow['DiscountedCashFlows'] = cashflow['UndiscountedCashFlows'] / (1 + rate)**cashflow.index
    cashflow['CumulativeDiscountedCashFlows'] = cashflow['DiscountedCashFlows'].cumsum()
    
    npv = cashflow['DiscountedCashFlows'].sum()
    
    
    neg_cumflow = cashflow[cashflow['CumulativeDiscountedCashFlows'].lt(0)]  # Use .lt() instead of <
    if len(neg_cumflow) == 0:
        discounted_payback_period = np.nan
    else:
        final_full_year = neg_cumflow.index.values[-1]  # Use [-1] to get last negative year
        try:
            fractional_yr = (-cashflow.loc[final_full_year, 'CumulativeDiscountedCashFlows'] / 
                           cashflow.loc[final_full_year + 1, 'DiscountedCashFlows'])
            discounted_payback_period = final_full_year + fractional_yr
        except (KeyError, ZeroDivisionError):
            discounted_payback_period = np.nan

    
    try:
        irr = npf.irr(cashflow['UndiscountedCashFlows']) * 100
    except:
        irr = np.nan

    return npv, discounted_payback_period, irr

def LCOE(energy, capex_t0, opex, inverter_cost, rate, land_cost, inverter_years=inverter_replacement_years, years=operation_period, dep=inverter_depreciation ):
    
    lcoe_cost=[]
    lcoe_energy=[]
    for i in range(years + 1):
        if i == 0:
            lcoe_cost.append(capex_t0+land_cost)
            lcoe_energy.append(0)
        elif i in inverter_years:
            lcoe_cost.append((opex + inverter_cost))
            lcoe_energy.append(energy.iloc[i-1])
        elif i ==years:
            lcoe_cost.append(-1* inverter_cost * (1 - dep) - land_cost)
            lcoe_energy.append(energy.iloc[i-1])
        else:
            lcoe_cost.append(opex)
            lcoe_energy.append(energy.iloc[i-1])
            
    lcoe_energy=pd.Series(lcoe_energy)
    lcoe_cost=pd.Series(lcoe_cost)
    lcoe_num = lcoe_cost/(1+rate)**lcoe_cost.index
    lcoe_dom = lcoe_energy/(1+rate)**lcoe_cost.index
    sum_lcoe_num = lcoe_num.sum()
    sum_lcoe_dom = lcoe_dom.sum()
    lcoe=sum_lcoe_num/sum_lcoe_dom*1e6 # LCOE in USD/MWh
    return lcoe
    
# =============================================================================
# 
# Initialize DataFrames for results
cost_profile = pd.DataFrame(columns=['Unit'] + [f'Cost_Year_{y+2022}' for y in range(0, operation_period+1)])
yearly_revenue_profile = pd.DataFrame(columns=['Unit'] + [f'Revenue_Year_{y+2022}' for y in range(0, operation_period+1)])
hourly_revenue_profile = pd.DataFrame()  # Will store all hourly revenue data
energy_profile = pd.DataFrame(columns=['Unit'] + [f'AC_kWh_Year_{y+2022}' for y in range(1, operation_period+1)])
tea_results = pd.DataFrame(columns=['Unit', 'Nominal Power Capacity [kW]', 'CAPEX_t0', 'Land Cost', 'NPV', 'IRR', 'DPBP', 'LCOE'])
capacity_factor_profile = pd.DataFrame(columns=['Unit'] + [f'CF_Year_{y+2022}' for y in range(1, operation_period+1)])

# =============================================================================
# Main analysis loop
for index, row in locations.iterrows():
    name = row['unit']  
    latitude = row['lat']
    longitude = row['long']
    num_panels = row['no_panels']
    acreage = row['acreage']
    

    nominal_capacity_kwdc = (num_panels * Pmax_STC) / 1000  # kWdc
    LAND_COST = acreage * land_price / 10000
    CAPEX_t0 = capex_t0 * nominal_capacity_kwdc
    OPEX = opex * nominal_capacity_kwdc
    INV_COST = inverter_cost * nominal_capacity_kwdc
    
    
    unit_hourly_energy = ac_results[[f'total_ac_output_[wh]_{name}']]
    AC = yearly_ac[f'total_ac_output_[wh]_{name}']
    AC_kWh = AC / 1000
    
    hours_per_year = 8760  
    cf_values = AC_kWh.values / (nominal_capacity_kwdc * hours_per_year)
    cf_row = [name] + list(cf_values)
    capacity_factor_profile.loc[index] = cf_row

    total_energy_MWh = AC_kWh.sum() / 1000
    
    
    COST = costflow(CAPEX_t0, OPEX, INV_COST, LAND_COST)
    
    
    yearly_REVENUE, hourly_REVENUE = revenueflow(
        unit_hourly_energy, 
        INV_COST, 
        LAND_COST, 
        full_hourly_prices,
        cel_rate,
        cel_price
    )
    
    
    hourly_REVENUE = hourly_REVENUE.to_frame(name)
    if hourly_revenue_profile.empty:
        hourly_revenue_profile = hourly_REVENUE
    else:
        hourly_revenue_profile = hourly_revenue_profile.join(hourly_REVENUE, how='outer')
    
    
    unit_npv, unit_dpbp, unit_irr = calculate_TEA_metrics(COST, yearly_REVENUE, discount_rate)
    unit_lcoe = LCOE(AC, CAPEX_t0, OPEX, INV_COST, discount_rate, LAND_COST)
    
    
    cost_profile.loc[index] = [name] + list(COST.values)
    yearly_revenue_profile.loc[index] = [name] + list(yearly_REVENUE.values)
    energy_profile.loc[index] = [name] + list(AC_kWh.values)
    
    unit_results = {
        'Unit': name,
        'Nominal Power Capacity [kW]': nominal_capacity_kwdc,
        'CAPEX_t0': CAPEX_t0,
        'Land Cost': LAND_COST,
        'NPV': unit_npv,
        'IRR': unit_irr,
        'DPBP': unit_dpbp,
        'LCOE': unit_lcoe,
    }
    tea_results.loc[index] = unit_results

# =============================================================================
# Entire System Calculations
Valsequillo_nominal_capacity = locations['no_panels'].sum() * Pmax_STC / 1000  # kW
VNC = Valsequillo_nominal_capacity / 1000  # MW

# Cost components
Valsequillo_module = Valsequillo_nominal_capacity * module
Valsequillo_bos_hardware = Valsequillo_nominal_capacity * bos_hardware
Valsequillo_installation = Valsequillo_nominal_capacity * installation
Valsequillo_soft_costs = Valsequillo_nominal_capacity * soft_costs
Valsequillo_inverter_cost = Valsequillo_nominal_capacity * inverter_cost

Valsequillo_capex_t0 = (Valsequillo_module + Valsequillo_bos_hardware + 
                        Valsequillo_installation + Valsequillo_inverter_cost + 
                        Valsequillo_soft_costs)
Valsequillo_opex = Valsequillo_nominal_capacity * opex
Valsequillo_land = locations['acreage'].sum() / 10000

# Aggregate energy and revenue
Valsequillo_energy = energy_profile.iloc[:, 1:].sum(axis=0).to_frame().T*1000 
Valsequillo_hourly_energy = ac_results.sum(axis=1)


# System-wide financial calculations
Valsequillo_cost = costflow(Valsequillo_capex_t0, Valsequillo_opex, Valsequillo_inverter_cost, Valsequillo_land*land_price)
Valsequillo_yearly_revenue, Valsequillo_hourly_revenue = revenueflow(
    Valsequillo_hourly_energy.to_frame(),
    Valsequillo_inverter_cost,
    Valsequillo_land*land_price,
    full_hourly_prices,
    cel_rate,
    cel_price
)

Valsequillo_npv, Valsequillo_dpbp, Valsequillo_irr = calculate_TEA_metrics(
    Valsequillo_cost, 
    Valsequillo_yearly_revenue, 
    discount_rate
)

Valsequillo_lcoe = LCOE(
    Valsequillo_energy.iloc[0], 
    Valsequillo_capex_t0, 
    Valsequillo_opex, 
    Valsequillo_inverter_cost, 
    discount_rate, 
    Valsequillo_land*land_price
)

# Store system-wide results
Valsequillo_results = {
    'Land Acreage [ha]': Valsequillo_land,
    'CAPEX [USD]': Valsequillo_capex_t0,
    'Land Cost[USD]': Valsequillo_land*land_price,
    'OPEX [USD/year]': Valsequillo_opex,
    'NPV [USD]': Valsequillo_npv,
    'LCOE [USD/MWh]': Valsequillo_lcoe,
    'IRR [%]': Valsequillo_irr,
    'DPBP [years]': Valsequillo_dpbp
}

first_year_energy = energy_profile.iloc[:, 1:].sum(axis=0).iloc[0] * 1000


# =============================================================================
# Sensitivity Analysis 
# =============================================================================

# Define parameter ranges
sensitivity_range = 0.25  # +/- 25% variation
params = {
    'module': {'values': np.linspace((1-sensitivity_range)*module, (1+sensitivity_range)*module, 20), 
               'label': 'Module Cost [USD/kW]', 'color': 'blue', 'type': 'capex'},
    'bos_hardware': {'values': np.linspace((1-sensitivity_range)*bos_hardware, (1+sensitivity_range)*bos_hardware, 20),
                    'label': 'BOS Hardware [USD/kW]', 'color': 'green', 'type': 'capex'},
    'installation': {'values': np.linspace((1-sensitivity_range)*installation, (1+sensitivity_range)*installation, 20),
                    'label': 'Installation [USD/kW]', 'color': 'red', 'type': 'capex'},
    'soft_costs': {'values': np.linspace((1-sensitivity_range)*soft_costs, (1+sensitivity_range)*soft_costs, 20),
                  'label': 'Soft Costs [USD/kW]', 'color': 'purple', 'type': 'capex'},
    'opex': {'values': np.linspace((1-sensitivity_range)*opex, (1+sensitivity_range)*opex, 20),
            'label': 'OPEX [USD/kW/year]', 'color': 'orange', 'type': 'opex'},
    'land_cost': {'values': np.linspace((1-sensitivity_range)*land_price, (1+sensitivity_range)*land_price, 20),
                 'label': 'Land Cost [USD]', 'color': 'brown', 'type': 'capex'},
    'energy_price_cf': {'values': np.linspace(1-sensitivity_range, 1+sensitivity_range, 20),
                       'label': 'Energy Price Multiplier', 'color': 'pink', 'type': 'financial'},
    'cel_price': {'values': np.linspace((1-sensitivity_range)*cel_price, (1+sensitivity_range)*cel_price, 20),
                 'label': 'CEL Price [USD/CEL]', 'color': 'cyan', 'type': 'financial'},
    'cel_rate': {'values': np.linspace((1-sensitivity_range)*cel_rate, (1+sensitivity_range)*cel_rate, 20),
                'label': 'CEL Allocation [CEL/MWh]', 'color': 'teal', 'type': 'financial'},
    'discount_rate': {'values': np.linspace((1-sensitivity_range)*discount_rate, (1+sensitivity_range)*discount_rate, 20),
                     'label': 'discount Rate [%]', 'color': 'gray', 'type': 'financial'},
}

# Initialize results storage
SAresults = {
    'LCOE [USD/MWh]': {param: [] for param in params},
    'DPBP [years]': {param: [] for param in params},
    'NPV [million USD]': {param: [] for param in params},
    'IRR [%]': {param: [] for param in params}
}

for param, config in params.items():
    print(f"Running sensitivity for {param}...")

    for value in config['values']:
        
        current_prices = full_hourly_prices.copy()
        current_cel_price = cel_price
        current_cel_rate = cel_rate
        current_ir = discount_rate
        current_land_cost = Valsequillo_land * land_price
        current_capex = Valsequillo_capex_t0
        current_opex = Valsequillo_opex

        
        if param == 'energy_price_cf':
            current_prices['Price'] *= value
        elif param == 'cel_price':
            current_cel_price = value
        elif param == 'cel_rate':
            current_cel_rate = value
        elif param == 'discount_rate':
            current_ir = value

        
        if param in ['module', 'bos_hardware', 'installation', 'soft_costs']:
            component_cost = value * Valsequillo_nominal_capacity
            if param == 'module':
                current_capex = (component_cost + Valsequillo_bos_hardware +
                                 Valsequillo_installation + Valsequillo_inverter_cost +
                                 Valsequillo_soft_costs)
            elif param == 'bos_hardware':
                current_capex = (Valsequillo_module + component_cost +
                                 Valsequillo_installation + Valsequillo_inverter_cost +
                                 Valsequillo_soft_costs)
            elif param == 'installation':
                current_capex = (Valsequillo_module + Valsequillo_bos_hardware +
                                 component_cost + Valsequillo_inverter_cost +
                                 Valsequillo_soft_costs)
            elif param == 'soft_costs':
                current_capex = (Valsequillo_module + Valsequillo_bos_hardware +
                                 Valsequillo_installation + Valsequillo_inverter_cost +
                                 component_cost)
        elif param == 'opex':
            current_opex = value * Valsequillo_nominal_capacity
        elif param == 'land_cost':
            current_land_cost = value * Valsequillo_land

        
        cost = costflow(current_capex, current_opex, Valsequillo_inverter_cost, current_land_cost)
        energy_input = Valsequillo_hourly_energy


        
        yearly_revenue, _ = revenueflow(
            energy_input.to_frame(),
            Valsequillo_inverter_cost,
            current_land_cost,
            current_prices,
            current_cel_rate,
            current_cel_price
        )

        
        yearly_energy = energy_input.groupby(energy_input.index.year).sum()

        lcoe = LCOE(
            yearly_energy,
            current_capex,
            current_opex,
            Valsequillo_inverter_cost,
            current_ir,
            current_land_cost
        )

        npv, dpbp, irr = calculate_TEA_metrics(cost, yearly_revenue, current_ir)

        
        SAresults['LCOE [USD/MWh]'][param].append(lcoe)
        SAresults['DPBP [years]'][param].append(dpbp)
        SAresults['NPV [million USD]'][param].append(npv / 1e6)
        SAresults['IRR [%]'][param].append(irr)

records = []

for metric, param_dict in SAresults.items():
    for param, values in param_dict.items():
        for i, v in enumerate(values):
            input_val = params[param]['values'][i]
            records.append({
                'Parameter': param,
                'Input Value': input_val,
                'Metric': metric,
                'Result': v
            })

SAresults_df = pd.DataFrame(records)

# =============================================================================
# Tornado Plot Visualization (2x2 Grid)
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt


param_order = [
    'module', 'bos_hardware', 'installation', 'soft_costs',
    'opex', 'land_cost','discount_rate','energy_price_cf', 'cel_price', 'cel_rate']

# Plotting function
def plot_tornado(ax, metric_name, results, baseline):
    param_labels = []
    left_values = []
    right_values = []
    colors = []

    for param in param_order:
        values = results[param]
        val_low = values[0]
        val_high = values[-1]
        param_labels.append(params[param]['label'])

        color = params[param]['color']
        if color == 'black':
            color = 'teal'  # Better visibility

        left_values.append(min(val_low, val_high))
        right_values.append(max(val_low, val_high))
        colors.append(color)

    y_pos = np.arange(len(param_labels))
    delta = [r - l for r, l in zip(right_values, left_values)]

    bars = ax.barh(y_pos, delta, left=left_values, color=colors, edgecolor='black', height=0.6)

    # Add labels at the ends
    span = max(right_values + left_values) - min(right_values + left_values)
    offset = 0.02 * span
    for i, (left, right) in enumerate(zip(left_values, right_values)):
        ax.text(left - offset, i, f"{left:.1f}", va='center', ha='right', fontsize=9)
        ax.text(right + offset, i, f"{right:.1f}", va='center', ha='left', fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_labels, fontsize=10, fontweight='bold')
    ax.axvline(baseline, color='gray', linestyle='--', linewidth=1)
    ax.set_title(metric_name, fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel("Metric Value", fontsize=11)

    all_vals = left_values + right_values
    min_val, max_val = min(all_vals), max(all_vals)
    span = max_val - min_val
    ax.set_xlim(min_val - 0.1 * span, max_val + 0.1 * span)

# Metrics to plot (exclude DPBP)
metrics = ['NPV [million USD]', 'LCOE [USD/MWh]', 'IRR [%]']

# Baselines: use mid-point of each parameter's range (can adjust if needed)
baseline_vals = {
    metric: np.median([SAresults[metric][param][10] for param in param_order])
    for metric in metrics
}

# Plotting 3x1
fig, axs = plt.subplots(3, 1, figsize=(14, 16))

for ax, metric in zip(axs, metrics):
    plot_tornado(ax, metric, SAresults[metric], baseline_vals[metric])

fig.suptitle(
    f"Sensitivity Analysis of Techno-Economic Metrics for a {VNC:.0f} MW {PVsystemconfig} with {surface_tilt}° Tilt",
    fontsize=18, fontweight='bold'
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# =============================================================================
# Monte Carlo Simulation
# =============================================================================

import pandas as pd
import numpy as np
from random import betavariate


num_simulations = 10000


global_cost_data = pd.read_csv(f"{PVsystemconfig}_cost.csv", encoding='latin1')
global_cost_data.set_index(global_cost_data.columns[0], inplace=True)
global_cost_data_numeric = global_cost_data.apply(pd.to_numeric, errors='coerce')

global_cost_summary = pd.DataFrame(index=global_cost_data_numeric.index)
global_cost_summary['Min'] = global_cost_data_numeric.min(axis=1)
global_cost_summary['Max'] = global_cost_data_numeric.max(axis=1)
global_cost_summary['Most_Likely'] = global_cost_data_numeric['Mexico']


other_inputs = pd.DataFrame({
    'OPEX [USD/kW]': [opex],
    'land_price [USD/ha]': [land_price],
    'discount_rate [%]': [discount_rate * 100],
    'cel_price [USD/CEL]': [cel_price],
    'cel_rate [CEL/MWh]': [cel_rate],
    'energy_price_cf': [1.0]  
}).T.rename(columns={0: 'Most_Likely'})


other_inputs['Min'] = other_inputs['Most_Likely']
other_inputs['Max'] = other_inputs['Most_Likely']


other_inputs.loc['land_price [USD/ha]', ['Min', 'Max']] *= [1 - sensitivity_range, 1 + sensitivity_range]
other_inputs.loc['energy_price_cf', ['Min', 'Max']] *= [1 - sensitivity_range, 1 + sensitivity_range]
other_inputs.loc['cel_rate [CEL/MWh]', ['Min', 'Max']] *= [1 - sensitivity_range, 1 + sensitivity_range]
other_inputs.loc['cel_price [USD/CEL]', ['Min', 'Max']] *= [1 - sensitivity_range, 1 + sensitivity_range]
other_inputs.loc['OPEX [USD/kW]', ['Min', 'Max']] = [opex_lower, opex_upper]
other_inputs.loc['discount_rate [%]', ['Min', 'Max']] = [discount_rate_lower * 100, discount_rate_upper * 100]


total_inputs = pd.concat([global_cost_summary, other_inputs])

# PERT distribution sampling
def pert_sample(a, b, c, lamb=4):
    r = c - a
    if r == 0:
        return a
    alpha = 1 + lamb * (b - a) / r
    beta = 1 + lamb * (c - b) / r
    return a + betavariate(alpha, beta) * r

# Generate samples
MC_samples = {
    param: [pert_sample(total_inputs.loc[param, 'Min'], total_inputs.loc[param, 'Most_Likely'], total_inputs.loc[param, 'Max'])
            for _ in range(num_simulations)]
    for param in total_inputs.index
}


MCresults = []

for i in range(num_simulations):
    print(f"current MC simulation {i}")
    # CAPEX breakdown
    module_i = MC_samples['Modules'][i] * Valsequillo_nominal_capacity
    bos_i = MC_samples['BoS Hardware'][i] * Valsequillo_nominal_capacity
    installation_i = MC_samples['Installation'][i] * Valsequillo_nominal_capacity
    soft_i = MC_samples['Soft costs'][i] * Valsequillo_nominal_capacity
    capex_i = module_i + bos_i + installation_i + soft_i + Valsequillo_inverter_cost
    opex_i = MC_samples['OPEX [USD/kW]'][i] * Valsequillo_nominal_capacity
    land_i = MC_samples['land_price [USD/ha]'][i] * Valsequillo_land
    irate_i = MC_samples['discount_rate [%]'][i] / 100
    deg_i = degradation_rate
    cel_price_i = MC_samples['cel_price [USD/CEL]'][i]
    cel_rate_i = MC_samples['cel_rate [CEL/MWh]'][i]
    price_cf_i = MC_samples['energy_price_cf'][i]
    degraded_hourly = []
    for year in range(1, operation_period + 1):
        degradation_factor = (1 - deg_i) ** (year - 1)
        year_mask = Valsequillo_hourly_energy.index.year == (base_year + year)
        year_energy = Valsequillo_hourly_energy[year_mask] * degradation_factor
        degraded_hourly.append(year_energy)

    energy_i_df = pd.concat(degraded_hourly).sort_index().to_frame()

    
    adjusted_prices = full_hourly_prices.copy()
    adjusted_prices['Price'] *= price_cf_i

   
    cost_i = costflow(capex_i, opex_i, Valsequillo_inverter_cost, land_i)
    revenue_i, _ = revenueflow(
        energy_i_df, Valsequillo_inverter_cost, land_i, adjusted_prices, cel_rate_i, cel_price_i
    )
    npv_i, dpbp_i, irr_i = calculate_TEA_metrics(cost_i, revenue_i, irate_i)
    annual_energy = energy_i_df.groupby(energy_i_df.index.year).sum()
    lcoe_i = LCOE(annual_energy, capex_i, opex_i, Valsequillo_inverter_cost, irate_i, land_i)
    if isinstance(lcoe_i, pd.Series):
        lcoe_i = lcoe_i.iloc[0]


    MCresults.append([npv_i, dpbp_i, irr_i, lcoe_i])


MCresults_df = pd.DataFrame(MCresults, columns=['NPV', 'DPBP', 'IRR', 'LCOE'])

# Success Factor
npv_success = (MCresults_df['NPV'] > 0)
irr_success = (MCresults_df['IRR'] > discount_rate * 100)

MCresults_df['Success (NPV>0)'] = npv_success
MCresults_df['IRR > discount rate'] = irr_success

npv_success_rate = npv_success.sum() / npv_success.count() * 100
irr_success_rate = irr_success.sum() / irr_success.count() * 100

print(f"✅ Success Probability (NPV > 0): {npv_success_rate:.2f}%")
print(f"✅ IRR > discount Rate ({discount_rate*100:.2f}%): {irr_success_rate:.2f}%")

MCsummary_stats = MCresults_df.describe(percentiles=[0.05, 0.5, 0.95])
print(MCsummary_stats)

# MC Visualization  
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 12))

metrics = ['NPV', 'LCOE', 'IRR']  # 3-row layout
for i, metric in enumerate(metrics):
    plt.subplot(3, 1, i + 1)
    sns.histplot(MCresults_df[metric], bins=50, kde=True, color='steelblue')

    # Plot statistics
    mean_val = MCresults_df[metric].mean()
    median_val = MCresults_df[metric].median()
    p5 = MCresults_df[metric].quantile(0.05)
    p95 = MCresults_df[metric].quantile(0.95)

    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle=':', label=f'Median: {median_val:.2f}')
    plt.axvline(p5, color='orange', linestyle='-.', label=f'5th percentile: {p5:.2f}')
    plt.axvline(p95, color='purple', linestyle='-.', label=f'95th percentile: {p95:.2f}')

    plt.title(f'Distribution of {metric}', fontsize=14, fontweight='bold')
    plt.xlabel(metric)
    plt.ylabel('Frequency')
    plt.legend()

plt.tight_layout()
plt.suptitle(f"Monte Carlo Distributions for Techno-Economic Metrics of {VNC:.0f} MW {PVsystemconfig} with {surface_tilt}° tilt", 
             fontsize=18, fontweight='bold', y=1.02)
plt.show()



# Prepare full input-output DataFrame
MCinputs_df = pd.DataFrame({
    'Modules': MC_samples['Modules'],
    'BoS Hardware': MC_samples['BoS Hardware'],
    'Installation': MC_samples['Installation'],
    'Soft Costs': MC_samples['Soft costs'],
    'OPEX': MC_samples['OPEX [USD/kW]'],
    'Land Cost': MC_samples['land_price [USD/ha]'],
    'Energy Price CF': MC_samples['energy_price_cf'],
    'CEL Price': MC_samples['cel_price [USD/CEL]'],
    'CEL Rate': MC_samples['cel_rate [CEL/MWh]'],
    'discount Rate': MC_samples['discount_rate [%]'],
})

# Combine inputs and outputs
full_MC_df = pd.concat([MCinputs_df, MCresults_df[['NPV', 'DPBP' ,'IRR', 'LCOE']]], axis=1)
full_MC_df = pd.concat([MCinputs_df, MCresults_df], axis=1)



# =============================================================================


Valsequillo_results_df = pd.DataFrame.from_dict(Valsequillo_results, orient='index', columns=['Value'])



# Save to Excel
with pd.ExcelWriter(f"Results_Summary_{VNC:.0f}MW_{PVsystemconfig}_{surface_tilt}deg.xlsx") as writer:
    # Summary sheet
    Valsequillo_results_df.to_excel(writer, sheet_name="Summary")
    
    # TEA Results 
    tea_results.to_excel(writer, sheet_name="TEA Results")
    
    # Sensitivity Analysis (flattened)
    SAresults_df.to_excel(writer, sheet_name="Sensitivity")
    
    # Cash flows
    cost_profile.to_excel(writer, sheet_name="Costs")
    hourly_revenue_profile.to_excel(writer, sheet_name="Revenues")
    
    # Energy
    ac_results.to_excel(writer, sheet_name="Hourly_Energy_Wh")
    energy_profile.to_excel(writer, sheet_name="Yearly_Energy_kWh")
    capacity_factor_profile.to_excel(writer, sheet_name="Capacity_Factor")
    # Monte Carlo
    MCsummary_stats.to_excel(writer, sheet_name="MC_Stats")
    full_MC_df.to_excel(writer, sheet_name="Full_MC_data")
    
    # Add metadata
    pd.DataFrame({
        'System': [f"{VNC:.0f} MW {PVsystemconfig}"],
        'Tilt': [f"{surface_tilt}°"],
        'Simulations': [num_simulations],
        'NPV > 0 rate':npv_success_rate,
        'IRR > Discount Rate': irr_success_rate
    }).to_excel(writer, sheet_name="Metadata", index=False)
# =============================================================================
