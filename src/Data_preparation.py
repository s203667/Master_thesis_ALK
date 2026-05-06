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
                 wind_file_name: str, fcr_file_name: str, imbalance_file_name: str):
        
        self.wind_file_name = wind_file_name
        self.el_price_file_name = el_price_file_name
        self.fcr_file_name = fcr_file_name
        self.imbalance_file_name = imbalance_file_name
        self.start_date = start_date
        self.end_date = end_date
        
        self.data = self.build()
    
    
    
    def prepare_wind_data(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "cache"))
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "wind_DK1_15min_scaled_210MW.csv")

        # --- LOAD CACHE ---
        if os.path.exists(cache_file):
            print("Loading cached DK1 wind data...")
            df = pd.read_csv(cache_file, parse_dates=["time"], index_col="time")
            return df.loc[self.start_date:self.end_date]

        print("Processing raw DK1 wind data...")

        # --- LOAD RAW CSV ---
        df = load_csv_data(self.wind_file_name, sep=';', decimal=',')

        # --- PARSE TIMESTAMP ---
        df["time"] = pd.to_datetime(df["Minutes5DK"])
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)

        # --- RAW OFFSHORE PRODUCTION (MW) ---
        df["wind_MW"] = df["OffshoreWindPower"]
        

        # --- SCALE TO 210 MW FARM ---
        DK1_capacity = 1600.8   # MW
        farm_capacity = 210.0   # MW
        scale_factor = farm_capacity / DK1_capacity

        df["wind_kW"] = df["wind_MW"] * scale_factor * 1000.0

        # --- RESAMPLE 5-min → 15-min BY AVERAGING ---
        df_15 = df["wind_kW"].resample("15min").mean()

        # --- CLEAN INVALID WIND DATA ---
        invalid_negatives = (df_15 < 0).sum()
        missing_values = df_15.isna().sum()
        if invalid_negatives or missing_values:
            print(f"Cleaning wind data: {invalid_negatives} negative values and {missing_values} missing values found.")

        # Wind power cannot be negative
        df_15 = df_15.clip(lower=0.0)

        # For gaps <= 1.5 hours (6 * 15-min intervals), fill with last known value
        df_15 = df_15.fillna(method='ffill', limit=6).fillna(method='bfill', limit=6)

        # For longer gaps, interpolate linearly between known values
        df_15 = df_15.interpolate(method='linear')

        # If any NaNs remain (e.g., at edges), fill with constant
        df_15 = df_15.fillna(method='ffill').fillna(method='bfill')

        # --- SAVE CACHE ---
        df_15.to_frame(name="wind_kW").to_csv(cache_file)
        print("Wind cache saved.")

        return df_15.loc[self.start_date:self.end_date].to_frame(name="wind_kW")

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

        # Calculate mean FCR price
        
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
