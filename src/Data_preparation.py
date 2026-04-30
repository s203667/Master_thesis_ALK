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
                 wind_file_name: str, fcr_file_name: str, year_mapping: dict = None):
        
        self.wind_file_name = wind_file_name
        self.el_price_file_name = el_price_file_name
        self.fcr_file_name = fcr_file_name
        self.start_date = start_date
        self.end_date = end_date
        # year_mapping: {wind_year: fcr_year} e.g. {2016: 2024, 2017: 2025}
        self.year_mapping = year_mapping or {}
        
        self.data = self.build()
    
    
    
    def prepare_wind_data(self):
        # --- CACHE PATH --- #
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))  #
        cache_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "cache"))  # 
        os.makedirs(cache_dir, exist_ok=True)  # 
        cache_file = os.path.join(cache_dir, "wmr_15min_cached.csv")  # 

        # --- LOAD CACHE IF EXISTS --- #
        if os.path.exists(cache_file):  # 
            print("Loading cached WMR data...")  # 
            df = pd.read_csv(cache_file, parse_dates=["time"], index_col="time")  # 
            return df.loc[self.start_date:self.end_date]  # 


        print("Processing raw WMR data...")  # 
        data = pd.DataFrame()

        # --- 2016 JAN–JUN --- #
        df = load_csv_data(  # 
            "Ørsted Confidential Information - SCADA_WMR_2016_JAN_JUN.csv",
            sep=";"
        )
        df["time"] = datetime(1900, 1, 1)  # 

        for sn in df["StationName"].unique():  # 
            mask = df["StationName"] == sn  # 
            df.loc[mask, "time"] = pd.date_range(  # 
                start=datetime(2016, 1, 1),
                freq="10min",
                periods=mask.sum()
            )

        data = pd.concat([data, df])  # 
        
        # --- REMAINING FILES --- #
        files_and_formats = [  # 
            ("Ørsted Confidential Information - SCADA_WMR_2016_JUL_DEC.csv", "%Y-%m-%d %H:%M:%S.%f"),
            ("Ørsted Confidential Information - SCADA_WMR_2017_JAN_JUN.csv", "%m/%d/%Y %H:%M:%S "),
            ("Ørsted Confidential Information - SCADA_WMR_2017_JUL_DEC.csv", "%Y-%m-%d %H:%M:%S.%f"),
        ]

        for file, fmt in files_and_formats:  # 
            df = load_csv_data(file)
            df["time"] = pd.to_datetime(df["TimeStamp"], format=fmt)
            data = pd.concat([data, df])


        # --- CLEAN --- #
        data.dropna(how="all", axis=0, inplace=True)
        data.dropna(how="all", axis=1, inplace=True)
        data.drop(columns=["TimeStamp"], inplace=True, errors="ignore")


        data = data[["time", "wtc_AcWindSp_mean", "wtc_ActPower_mean"]]  #

        data["time"] = pd.to_datetime(data["time"])
        data.set_index("time", inplace=True)
        data.sort_index(inplace=True)  #


        # --- INTERPOLATE TO 15 MIN --- #
        data_15min = data.resample("15min").mean()
        data_15min["wtc_ActPower_mean"] = (
            data_15min["wtc_ActPower_mean"]
            .interpolate(method="time")
            .ffill()
            .bfill()
)

        # Compute largest day-over-day power difference (96 timesteps = 24 hours in 15-min intervals)
        power = data_15min['wtc_ActPower_mean'].values
        if len(power) > 96:
            day_diffs = np.abs(power[96:] - power[:-96])
            max_day_diff = day_diffs.max()
            print(f"Largest day-over-day power difference: {max_day_diff}")

        # --- SAVE CACHE --- #
        data_15min.to_csv(cache_file)  # 
        print("Cache saved")  # 


        return data_15min.loc[self.start_date:self.end_date]

    
    def prepare_el_price_data(self):
        #cache
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 
        cache_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "cache"))  # 
        os.makedirs(cache_dir, exist_ok=True)  #
        cache_file = os.path.join(cache_dir, "el_price_15min_cached.csv")  #

        # --- LOAD CACHE IF EXISTS --- #
        if os.path.exists(cache_file):  #
            print("Loading cached electricity price data...")  #
            df = pd.read_csv(cache_file, parse_dates=["time"], index_col="time")  #
            return df.loc[self.start_date:self.end_date]  #
        
        # --- PROCESS RAW DATA --- #
        df_el_price = load_csv_data(self.el_price_file_name, sep=";", decimal=",")  #
        df_el_price["time"] = pd.to_datetime(df_el_price["HourDK"], format="%Y-%m-%d %H:%M:%S")  #
        df_el_price.set_index("time", inplace=True)  #
        df_el_price = df_el_price[~df_el_price.index.duplicated(keep='first')]  # Remove duplicates
        df_el_price.sort_index(inplace=True)  #

        # Select the price column (using EUR for consistency)
        df_el_price = df_el_price[["SpotPriceEUR"]].rename(columns={"SpotPriceEUR": "el_price[EUR/MWh]"})  #

        # Resample to 15-minute intervals by forward filling the hourly prices
        df_el_15min = df_el_price.resample("15min").ffill()  #

        # --- SAVE CACHE --- #
        df_el_15min.to_csv(cache_file)  #
        print("Electricity price cache saved!")  #

        return df_el_15min.loc[self.start_date:self.end_date]


    def _align_data_by_calendar_time(self, source_data, target_index, target_year):
        """
        Align source_data (e.g. FCR from 2024) to target_index (e.g. 2016)
        by shifting the year and forward/backward filling.
        """

        df = source_data.copy()

        # Shift year
        df.index = df.index.map(lambda t: t.replace(year=target_year))

        # Reindex
        df_aligned = df.reindex(target_index)

        # fill BOTH directions
        df_aligned = df_aligned.ffill().bfill()

        return df_aligned


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


    def build(self):
        """
        Build the complete dataset with proper year alignment.
        If year_mapping is provided (e.g., {2016: 2024, 2017: 2025}), 
        FCR data from specified years will be aligned by calendar time to wind/price data years.
        """
        wind_data = self.prepare_wind_data()  # Filtered by date range
        el_price_data = self.prepare_el_price_data()  # Filtered by date range
        FCR_data_all = self.prepare_FCR_data(filter_by_date=False)  # NOT filtered, load all years
        
        # Get unique years in wind data
        wind_years = sorted(wind_data.index.year.unique())
        print(f"Wind/Price data years: {wind_years}")
        print(f"Year mapping: {self.year_mapping}")
        
        # Align FCR data year by year
        aligned_fcr_parts = []
        
        for wind_year in wind_years:
            # Determine which FCR year to use for this wind year
            fcr_year = self.year_mapping.get(wind_year, wind_year)
            
            print(f"Aligning FCR data from {fcr_year} to wind/price data year {wind_year} by calendar time...")
            
            # Get wind data for this year
            wind_year_data = wind_data[wind_data.index.year == wind_year]
            
            # Get FCR data for the corresponding FCR year
            fcr_year_data = FCR_data_all[FCR_data_all.index.year == fcr_year]
            
            if len(fcr_year_data) == 0:
                print(f"WARNING: No FCR data found for year {fcr_year}")
                continue
            
            # Align FCR data to match wind data by calendar time (month-day-hour)
            aligned_fcr = self._align_data_by_calendar_time(fcr_year_data, wind_year_data.index, wind_year)
            aligned_fcr_parts.append(aligned_fcr)
        
        # Concatenate aligned FCR parts
        if aligned_fcr_parts:
            FCR_price_data = pd.concat(aligned_fcr_parts).sort_index()
            FCR_price_data = FCR_price_data.ffill().bfill()
            
        else:
            FCR_price_data = pd.DataFrame()

        if FCR_price_data.isna().any().any():
            raise ValueError("NaN values detected in FCR data after alignment!")


        # --- FINAL DF FORMAT --- #
        df = pd.DataFrame({  # 
            "time": wind_data.index,  # 
            "wind_speed[m/s]": wind_data["wtc_AcWindSp_mean"],  # 
            "wind_power[kW]": wind_data["wtc_ActPower_mean"],  # 
        })

        df = df.reset_index(drop=True)  # 

        # Return all prepared data
        return {
            'wind': wind_data,
            'el_price': el_price_data,
            'fcr': FCR_price_data
        }

if __name__ == "__main__":
    prep = DataPreparationCSV(
        start_date="2016-01-01",
        end_date="2017-12-31",
        el_price_file_name="Elspotprices_16-17.csv",
        wind_file_name="",
        fcr_file_name="FcrDK1_24-25.csv",
        year_mapping={2016: 2024, 2017: 2025}  # Align 2016→2024, 2017→2025
    )
    
    # Calculate and print largest day-over-day power difference
    wind_data = prep.data['wind']
    power = wind_data['wtc_ActPower_mean'].values
    if len(power) > 96:
        day_diffs = np.abs(power[96:] - power[:-96])
        max_day_diff = day_diffs.max()
        print(f"Largest day-over-day power difference: {max_day_diff}")


