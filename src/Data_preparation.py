from data_loader import load_csv_data
import pandas as pd
from datetime import datetime
import os

class Expando(object):
    pass
#


class DataPreparationCSV:
    
    def __init__(self, start_date: str, end_date: str, el_price_file_name: str,
                 wind_file_name: str):
        
        self.wind_file_name = wind_file_name
        self.el_price_file_name = el_price_file_name
        self.start_date = start_date
        self.end_date = end_date
        
        self.data = self.build()
    
    
    
    def prepare_wind_data(self):
        # --- CACHE PATH --- #
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # <<< REWRITE >>>
        cache_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "cache"))  # <<< REWRITE >>>
        os.makedirs(cache_dir, exist_ok=True)  # <<< REWRITE >>>
        cache_file = os.path.join(cache_dir, "wmr_15min_cached.csv")  # <<< REWRITE >>>

        # --- LOAD CACHE IF EXISTS --- #
        if os.path.exists(cache_file):  # <<< REWRITE >>>
            print("Loading cached WMR data...")  # <<< REWRITE >>>
            df = pd.read_csv(cache_file, parse_dates=["time"], index_col="time")  # <<< REWRITE >>>
            return df.loc[self.start_date:self.end_date]  # <<< REWRITE >>>


        print("Processing raw WMR data...")  # <<< REWRITE >>>
        data = pd.DataFrame()

        # --- 2016 JAN–JUN --- #
        df = load_csv_data(  # <<< REWRITE >>>
            "Ørsted Confidential Information - SCADA_WMR_2016_JAN_JUN.csv",
            sep=";"
        )
        df["time"] = datetime(1900, 1, 1)  # <<< REWRITE >>>

        for sn in df["StationName"].unique():  # <<< REWRITE >>>
            mask = df["StationName"] == sn  # <<< REWRITE >>>
            df.loc[mask, "time"] = pd.date_range(  # <<< REWRITE >>>
                start=datetime(2016, 1, 1),
                freq="10min",
                periods=mask.sum()
            )

        data = pd.concat([data, df])  # <<< REWRITE >>>
        
        # --- REMAINING FILES --- #
        files_and_formats = [  # <<< REWRITE >>>
            ("Ørsted Confidential Information - SCADA_WMR_2016_JUL_DEC.csv", "%Y-%m-%d %H:%M:%S.%f"),
            ("Ørsted Confidential Information - SCADA_WMR_2017_JAN_JUN.csv", "%m/%d/%Y %H:%M:%S "),
            ("Ørsted Confidential Information - SCADA_WMR_2017_JUL_DEC.csv", "%Y-%m-%d %H:%M:%S.%f"),
        ]

        for file, fmt in files_and_formats:  # <<< REWRITE >>>
            df = load_csv_data(file)
            df["time"] = pd.to_datetime(df["TimeStamp"], format=fmt)
            data = pd.concat([data, df])


        # --- CLEAN --- #
        data.dropna(how="all", axis=0, inplace=True)
        data.dropna(how="all", axis=1, inplace=True)
        data.drop(columns=["TimeStamp"], inplace=True, errors="ignore")


        data = data[["time", "wtc_AcWindSp_mean", "wtc_ActPower_mean"]]  # <<< REWRITE >>>

        data["time"] = pd.to_datetime(data["time"])
        data.set_index("time", inplace=True)
        data.sort_index(inplace=True)  # <<< REWRITE >>>


        # --- INTERPOLATE TO 15 MIN --- #
        data_15min = data.resample("15min").mean().interpolate(method="time")  # <<< REWRITE >>>


        # --- SAVE CACHE --- #
        data_15min.to_csv(cache_file)  # <<< REWRITE >>>
        print("Cache saved")  # <<< REWRITE >>>


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


    def prepare_FCR_data(self):
        # --- CACHE SETUP --- #
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "cache"))
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "fcr_4h_cached_v2.csv")

        # --- LOAD CACHE --- #
        if os.path.exists(cache_file):
            print("Loading cached FCR 4h data...")
            df = pd.read_csv(cache_file, parse_dates=["time"], index_col="time")
            return df.loc[self.start_date:self.end_date]

        print("Processing raw FCR data...")

        # --- LOAD RAW --- #
        df_fcr = load_csv_data("FcrReservesDK1.csv", sep=";", decimal=",")

        # Parse timestamp
        df_fcr["time"] = pd.to_datetime(df_fcr["HourDK"], format="%Y-%m-%d %H:%M:%S")
        df_fcr.set_index("time", inplace=True)
        df_fcr.sort_index(inplace=True)

        # Remove duplicates
        df_fcr = df_fcr[~df_fcr.index.duplicated(keep='first')]

        # --- SELECT RELEVANT COLUMNS --- #
        df_fcr = df_fcr[[
            "FCR_UpPriceEUR",
            "FCR_DownPriceEUR",
            "FCR_UpPurchased",
            "FCR_DownPurchased"
        ]].rename(columns={
            "FCR_UpPriceEUR": "price_FCR_up",
            "FCR_DownPriceEUR": "price_FCR_down",
            "FCR_UpPurchased": "volume_up",
            "FCR_DownPurchased": "volume_down"
        })

        # Calculate mean FCR price
        
        # 🔹 --- RESAMPLE DIRECTLY TO 4 HOURS --- #
        df_fcr_4h = df_fcr.resample("4H").mean()
        df_fcr_4h["price_FCR"] = (df_fcr_4h["price_FCR_up"] + df_fcr_4h["price_FCR_down"]) / 2

        # --- SAVE CACHE --- #
        df_fcr_4h.to_csv(cache_file)
        print("FCR 4h cache saved!")

        return df_fcr_4h.loc[self.start_date:self.end_date]


    def build(self):
        # Create a date range for the entire period
        #dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        
        
        wind_data = self.prepare_wind_data()
        el_price_data = self.prepare_el_price_data()
        FCR_price_data = self.prepare_FCR_data()

        print("Electricity price data head:\n", el_price_data.head())
        print("FCR price data head:\n", FCR_price_data.head())

        # --- FINAL DF FORMAT --- #
        df = pd.DataFrame({  # <<< REWRITE >>>
            "time": wind_data.index,  # <<< REWRITE >>>
            "wind_speed[m/s]": wind_data["wtc_AcWindSp_mean"],  # <<< REWRITE >>>
            "wind_power[kW]": wind_data["wtc_ActPower_mean"],  # <<< REWRITE >>>
        })

        df = df.reset_index(drop=True)  # <<< REWRITE >>>

        print("\nFinal DataFrame head:\n")  # <<< REWRITE >>>
        print(df.head())  # <<< REWRITE >>>

        
        

        return df  # <<< REWRITE >>>

if __name__ == "__main__":
    prep = DataPreparationCSV(
        start_date="2016-01-01",
        end_date="2017-12-31",
        el_price_file_name="",  # adjust as needed
        wind_file_name=""  # adjust as needed
    )

