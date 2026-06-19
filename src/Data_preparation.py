from data_loader import load_csv_data
import pandas as pd
import numpy as np
from datetime import datetime
import os

class Expando(object):
    pass
#


class DataPreparationCSV:
    
    def __init__(self, start_date: str, end_date: str, el_price_file_name: str,
                 wind_file_name: str, fcr_file_name: str, imbalance_file_name: str,
                 wind_source: str = "scada", wtg_file_name: str = "V164-8MW.wtg"):
        
        self.wind_file_name = wind_file_name
        self.el_price_file_name = el_price_file_name
        self.fcr_file_name = fcr_file_name
        self.imbalance_file_name = imbalance_file_name
        self.start_date = start_date
        self.end_date = end_date
        self.wind_source = wind_source
        self.wtg_file_name = wtg_file_name
    
        self.data = self.build()
    
    
    
    def prepare_wind_data(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "cache"))
        os.makedirs(cache_dir, exist_ok=True)

        # --- CACHE FILE NAME DEPENDS ON SOURCE ---
        cache_file = os.path.join(
            cache_dir,
            f"wind_DK1_15min_scaled_210MW_{self.wind_source}.csv"
        )

        if os.path.exists(cache_file):
            print(f"Loading cached wind data ({self.wind_source})...")
            df = pd.read_csv(cache_file, parse_dates=["time"], index_col="time")
            return df.loc[self.start_date:self.end_date]

        print(f"Processing raw wind data ({self.wind_source})...")

        if self.wind_source == "scada":
            df_15 = self._prepare_wind_scada()
        elif self.wind_source == "era5":
            df_15 = self._prepare_wind_era5()
        else:
            raise ValueError(f"Unknown wind_source '{self.wind_source}'. Use 'scada' or 'era5'.")

        df_15.to_frame(name="wind_kW").to_csv(cache_file)
        print(f"Wind cache saved ({self.wind_source}).")
        return df_15.loc[self.start_date:self.end_date].to_frame(name="wind_kW")


    def _prepare_wind_scada(self):
        """Load and scale Energinet 5-min SCADA offshore wind data to 210 MW farm."""
        df = load_csv_data(self.wind_file_name, sep=';', decimal=',')
        df["time"] = pd.to_datetime(df["Minutes5DK"])
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)

        DK1_capacity  = 1600.8   # MW — total DK1 offshore installed
        farm_capacity = 210.0    # MW — reference farm
        scale_factor  = farm_capacity / DK1_capacity

        df["wind_kW"] = df["OffshoreWindPower"] * scale_factor * 1000.0

        df_15 = df["wind_kW"].resample("15min").mean()
        df_15 = df_15.clip(lower=0.0)
        df_15 = df_15.interpolate(limit=1, limit_direction='both')
        df_15 = df_15.ffill().bfill()
        return df_15

    def _prepare_wind_era5(self):
        """Load ECMWF IFS hourly wind speeds, apply V164-8MW power curve,
        apply offshore loss factors, and upsample to 15-min."""
        import xml.etree.ElementTree as ET
        from scipy.interpolate import interp1d

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        data_dir  = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))

        # --- 1. Parse power curve from WAsP file ---
        wtg_path = os.path.join(BASE_DIR, "..", "data", "Vestas_V164-8MW.wtg")
        tree = ET.parse(wtg_path)
        root = tree.getroot()

        sss      = root.find('.//StartStopStrategy')
        v_cutin  = float(sss.get('LowSpeedCutIn'))
        v_cutout = float(sss.get('HighSpeedCutOut'))

        records = []
        for dp in root.findall('.//DataPoint'):
            ws  = float(dp.get('WindSpeed'))
            pwr = float(dp.get('PowerOutput')) / 1000.0   # W → kW
            records.append((ws, pwr))

        df_pc = pd.DataFrame(records, columns=['ws', 'pwr'])
        df_pc = df_pc.sort_values('ws').reset_index(drop=True)

        df_pc = pd.concat([
            pd.DataFrame({'ws': [0.0, v_cutin - 0.01], 'pwr': [0.0, 0.0]}),
            df_pc,
            pd.DataFrame({'ws': [v_cutout + 0.01], 'pwr': [0.0]})
        ], ignore_index=True).sort_values('ws').reset_index(drop=True)

        pc_interp = interp1d(
            df_pc['ws'], df_pc['pwr'],
            kind='linear', bounds_error=False, fill_value=0.0
        )

        rated_kw      = df_pc['pwr'].max()     # 8000 kW
        farm_capacity = 216_000.0              # kW
        n_turbines    = farm_capacity / rated_kw   # 26

        # --- 2. Offshore loss factors ---
        WAKE_LOSS         = 0.06   # 6% — typical large offshore array
        AVAILABILITY_LOSS = 0.05   # 5%  — offshore O&M downtime
        ELECTRICAL_LOSS   = 0.015  # 1.5% — array + export cable losses
        LOSS_FACTOR = (1 - WAKE_LOSS) * (1 - AVAILABILITY_LOSS) * (1 - ELECTRICAL_LOSS)

        print(f"  Power curve  : V164-8MW, rated {rated_kw:.0f} kW, "
            f"{n_turbines:.0f} turbines → {farm_capacity/1000:.0f} MW farm")
        print(f"  Loss factors : wake {WAKE_LOSS*100:.0f}%, "
            f"availability {AVAILABILITY_LOSS*100:.0f}%, "
            f"electrical {ELECTRICAL_LOSS*100:.1f}%")
        print(f"  Combined loss factor: {LOSS_FACTOR:.4f}  "
            f"({(1-LOSS_FACTOR)*100:.1f}% total loss)")

        # --- 3. Load ECMWF wind speed CSV ---
        ws_path = os.path.join(BASE_DIR, "..", "data", self.wind_file_name)

        with open(ws_path, 'r') as f:
            lines = f.readlines()
        header_row = next(i for i, l in enumerate(lines) if l.startswith('time'))

        df_ecmwf = pd.read_csv(ws_path, skiprows=header_row, parse_dates=['time'])
        df_ecmwf.columns = ['time', 'wind_speed_ms']
        df_ecmwf.set_index('time', inplace=True)

        df_ecmwf.index = (
            df_ecmwf.index
            .tz_localize('UTC')
            .tz_convert('Europe/Copenhagen')
            .tz_localize(None)
        )
        df_ecmwf.sort_index(inplace=True)
        df_ecmwf = df_ecmwf[~df_ecmwf.index.duplicated(keep='first')]

        # --- 4. Hub height correction: 100m → 103.6m ---
        Z_ECMWF, Z_HUB, Z0 = 100.0, 103.6, 0.0002
        correction = np.log(Z_HUB / Z0) / np.log(Z_ECMWF / Z0)
        df_ecmwf['wind_speed_hub'] = df_ecmwf['wind_speed_ms'] * correction

        # --- 5. Apply power curve, scale to farm, apply losses ---
        df_ecmwf['wind_kW_gross'] = (
            pc_interp(df_ecmwf['wind_speed_hub'].values) * n_turbines
        )
        df_ecmwf['wind_kW'] = (df_ecmwf['wind_kW_gross'] * LOSS_FACTOR).clip(lower=0.0)

        cf_gross = df_ecmwf['wind_kW_gross'].mean() / farm_capacity * 100
        cf_net   = df_ecmwf['wind_kW'].mean()       / farm_capacity * 100
        print(f"  ECMWF gross capacity factor : {cf_gross:.1f}%")
        print(f"  ECMWF net  capacity factor  : {cf_net:.1f}%  (after losses)")

        # --- 6. Upsample hourly → 15-min by linear interpolation ---
        df_15 = df_ecmwf[['wind_speed_hub', 'wind_kW_gross', 'wind_kW']].resample(
            '15min'
        ).interpolate(method='linear')
        df_15 = df_15.clip(lower=0.0)

        # Return only wind_kW so downstream code sees the same column name
        return df_15['wind_kW']

    def prepare_el_price_data(self):
        #cache
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "cache"))
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "el_price_15min_cached.csv")

        # --- LOAD CACHE IF EXISTS --- #
        if os.path.exists(cache_file):
            print("Loading cached electricity price data...")
            df = pd.read_csv(cache_file, parse_dates=["time"], index_col="time")
            return df.loc[self.start_date:self.end_date]

        # --- PROCESS RAW DATA --- #
        data_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
        elspot_path = os.path.join(data_dir, "Elspotprices.csv")
        dayahead_path = os.path.join(data_dir, "DayAheadPrices.csv")

        combined = []

        if os.path.exists(elspot_path):
            df_elspot = load_csv_data("Elspotprices.csv", sep=";", decimal=",")
            df_elspot["time"] = pd.to_datetime(df_elspot["HourDK"], format="%Y-%m-%d %H:%M:%S")
            df_elspot.set_index("time", inplace=True)
            df_elspot = df_elspot[~df_elspot.index.duplicated(keep='first')]
            df_elspot = df_elspot[["SpotPriceEUR"]].rename(columns={"SpotPriceEUR": "el_price[EUR/MWh]"})
            df_elspot = df_elspot.resample("15min").ffill()
            combined.append(df_elspot)

        if os.path.exists(dayahead_path):
            df_dayahead = load_csv_data("DayAheadPrices.csv", sep=";", decimal=",")
            df_dayahead["time"] = pd.to_datetime(df_dayahead["TimeDK"], format="%Y-%m-%d %H:%M:%S")
            df_dayahead.set_index("time", inplace=True)
            df_dayahead = df_dayahead[~df_dayahead.index.duplicated(keep='first')]
            df_dayahead = df_dayahead[["DayAheadPriceEUR"]].rename(columns={"DayAheadPriceEUR": "el_price[EUR/MWh]"})
            df_dayahead = df_dayahead.resample("15min").ffill()
            combined.append(df_dayahead)

        if not combined:
            raise FileNotFoundError("No electricity price source files found")

        df_el_price = pd.concat(combined).sort_index()
        df_el_price = df_el_price[~df_el_price.index.duplicated(keep='last')]
        df_el_price = df_el_price.fillna(method='ffill').fillna(method='bfill')
        

        # --- SAVE CACHE --- #
        df_el_price.to_csv(cache_file)
        print("Electricity price cache saved!")

        return df_el_price.loc[self.start_date:self.end_date]


   

    def prepare_FCR_data(self, filter_by_date=False):
        # --- CACHE SETUP --- #
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "cache"))
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "fcr_4h_cached_v2.csv")

        # --- LOAD CACHE --- #
        if os.path.exists(cache_file):
            print("Loading cached FCR 4h data...")
            df = pd.read_csv(cache_file, parse_dates=["time"], index_col="time")
            if filter_by_date:
                return df.loc[self.start_date:self.end_date]
            return df

        print("Processing raw FCR data...")

        # --- LOAD RAW --- #
        df_fcr = load_csv_data(self.fcr_file_name, sep=";", decimal=",")

        # Parse timestamp
        df_fcr["time"] = pd.to_datetime(df_fcr["HourDK"], format="%Y-%m-%d %H:%M:%S")
        df_fcr.set_index("time", inplace=True)
        df_fcr.sort_index(inplace=True)

        # Remove duplicates
        df_fcr = df_fcr[~df_fcr.index.duplicated(keep='first')]

        # --- SELECT RELEVANT COLUMNS --- # 
        df_fcr = df_fcr[[
            "FCRdomestic_MW",
            "FCRdk_EUR"
        ]].rename(columns={
            "FCRdomestic_MW": "FCR_volume",
            "FCRdk_EUR": "price_FCR"
        })

    
        # 🔹 --- RESAMPLE DIRECTLY TO 4 HOURS --- #
        df_fcr_4h = df_fcr.resample("4H").mean()
        df_fcr_4h["price_FCR"] = df_fcr_4h["price_FCR"] / 1000  # Convert from EUR/MW to EUR/kW
        df_fcr_4h["FCR_volume"] = df_fcr_4h["FCR_volume"] * 1000  # Convert from MW to kW
  

        # --- SAVE CACHE --- #
        df_fcr_4h.to_csv(cache_file)
        print("FCR 4h cache saved!")


        if filter_by_date:
            return df_fcr_4h.loc[self.start_date:self.end_date]
        return df_fcr_4h

    def prepare_imbalance_data(self):

        df_imb = load_csv_data(self.imbalance_file_name, sep=";", decimal=",")
        df_imb["time"] = pd.to_datetime(df_imb["TimeDK"])
        df_imb.set_index("time", inplace=True)
        df_imb.sort_index(inplace=True)

        return df_imb[["ImbalancePriceEUR"]]

    def build(self):
  
        wind_data = self.prepare_wind_data()  # Filtered by date range
        el_price_data = self.prepare_el_price_data()  # Filtered by date range
        FCR_price_data = self.prepare_FCR_data(filter_by_date=True)  # Filtered by date range
        imbalance_data = self.prepare_imbalance_data()  # Filtered by date range

        if FCR_price_data.isna().any().any():
            raise ValueError("NaN values detected in FCR data after alignment!")

        # Return all prepared data
        return {
            'wind': wind_data,
            'el_price': el_price_data,
            'fcr': FCR_price_data,
            'imbalance': imbalance_data
        }

# if __name__ == "__main__":
#     prep = DataPreparationCSV(
#         start_date="2025-03-05",
#         end_date="2026-03-05",
#         el_price_file_name="DayAheadPrices.csv",
#         wind_file_name="WindProd5MinRealtime.csv",
#         fcr_file_name="FcrDK1.csv"    )

#################### OLD CODE FOR OLD DATASET (NOT USED) ####################
       # --- 2016 JAN–JUN --- #
#         df = load_csv_data(  # 
#             "Ørsted Confidential Information - SCADA_WMR_2016_JAN_JUN.csv",
#             sep=";"
#         )
#         df["time"] = datetime(1900, 1, 1)  # 

#         for sn in df["StationName"].unique():  # 
#             mask = df["StationName"] == sn  # 
#             df.loc[mask, "time"] = pd.date_range(  # 
#                 start=datetime(2016, 1, 1),
#                 freq="10min",
#                 periods=mask.sum()
#             )

#         data = pd.concat([data, df])  # 
        
#         # --- REMAINING FILES --- #
#         files_and_formats = [  # 
#             ("Ørsted Confidential Information - SCADA_WMR_2016_JUL_DEC.csv", "%Y-%m-%d %H:%M:%S.%f"),
#             ("Ørsted Confidential Information - SCADA_WMR_2017_JAN_JUN.csv", "%m/%d/%Y %H:%M:%S "),
#             ("Ørsted Confidential Information - SCADA_WMR_2017_JUL_DEC.csv", "%Y-%m-%d %H:%M:%S.%f"),
#         ]

#         for file, fmt in files_and_formats:  # 
#             df = load_csv_data(file)
#             df["time"] = pd.to_datetime(df["TimeStamp"], format=fmt)
#             data = pd.concat([data, df])


#         # --- CLEAN --- #
#         data.dropna(how="all", axis=0, inplace=True)
#         data.dropna(how="all", axis=1, inplace=True)
#         data.drop(columns=["TimeStamp"], inplace=True, errors="ignore")


#         data = data[["time", "wtc_AcWindSp_mean", "wtc_ActPower_mean"]]  #

#         data["time"] = pd.to_datetime(data["time"])
#         data.set_index("time", inplace=True)
#         data.sort_index(inplace=True)  #


#         # --- INTERPOLATE TO 15 MIN --- #
#         data_15min = data.resample("15min").mean()
#         data_15min["wtc_ActPower_mean"] = (
#             data_15min["wtc_ActPower_mean"]
#             .interpolate(method="time")
#             .ffill()
#             .bfill()
# )

#         # Compute largest day-over-day power difference (96 timesteps = 24 hours in 15-min intervals)
#         power = data_15min['wtc_ActPower_mean'].values
#         if len(power) > 96:
#             day_diffs = np.abs(power[96:] - power[:-96])
#             max_day_diff = day_diffs.max()
#             print(f"Largest day-over-day power difference: {max_day_diff}")

#  def _align_data_by_calendar_time(self, source_data, target_index, target_year):
#         """
#         Align source_data (e.g. FCR from 2024) to target_index (e.g. 2016)
#         by shifting the year and forward/backward filling.
#         """

#         df = source_data.copy()

#         # Shift year
#         df.index = df.index.map(lambda t: t.replace(year=target_year))

#         # Reindex
#         df_aligned = df.reindex(target_index)

#         # fill BOTH directions
#         df_aligned = df_aligned.ffill().bfill()

#         return df_aligned
