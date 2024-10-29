# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

import wget
import zipfile
import pandas as pd
import math
import re
import numpy as np

import os
import urllib.request
import zipfile

def downloadAndExtract(
    url="https://api.eia.gov/bulk/EBA.zip", 
    destination_dir=None,
    cluster=False
):
    # Determine the destination directory based on the file's location if not provided
    if destination_dir is None:
        try:
            # Try to use __file__ to get the directory of this script
            destination_dir = os.path.join(os.path.dirname(__file__), "EBA")
        except NameError:
            # Fallback to current working directory in interactive environments
            destination_dir = os.path.join(os.getcwd(), "EBA")
    
    # Check if destination directory already exists
    if os.path.exists(destination_dir):
        print(f"The directory '{destination_dir}' already exists. Data may have already been downloaded.")
        return

    # If running on a cluster, print instructions for manual download
    if cluster:
        print("If downloading from a cluster, you may need to do so from Terminal. "
              "Paste the following into a Terminal to manually download and extract:")
        print(f"""
        mkdir -p "{destination_dir}" && \
        wget -O eba_data.zip {url} && \
        unzip eba_data.zip -d "{destination_dir}" && \
        rm eba_data.zip
        """)
        return
    
    # Download the file
    zip_path = "eba_data.zip"
    urllib.request.urlretrieve(url, zip_path)
    print("Download complete.")
    
    # Extract the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination_dir)
    print(f"Data extracted to '{destination_dir}'.")

    # Cleanup: remove the downloaded zip file
    os.remove(zip_path)



eba_json = None
ba_list = []
ts_list = []

def prepareEIAData(EIA_data_path):
    global eba_json
    global ba_list
    global ts_list

    # EBA.txt includes time series for power generation from
    # each balancing authority in json format.
    #
    eba_json = pd.read_json("{0}/EBA.txt".format(EIA_data_path), lines=True)
    #writeCSV(eba_json)

    # Construct list of BAs (ba_list)
    # Construct list of time series (ts_list) using CISO as reference
    #
    series_id_unique = list(eba_json.series_id.unique())
    series_id_unique = list(filter(lambda x: type(x) == str, series_id_unique))
    
    ba_num = 0
    for sid in series_id_unique:
        m = re.search("EBA.(.+?)-", str(sid))
        ba_this = m.group(1)
        if ba_this not in ba_list:
            ba_list.append(ba_this)
            ba_num = ba_num + 1

        if ba_this == "CISO":
            m = re.search("EBA.CISO-([A-Z\-]+\.)([A-Z\.\-]*)", str(sid))
            ts_list.append(m.group(2))
    print("EIA data prep done!")

    return eba_json, ba_list, ts_list


# Energy types
ng_list = [
    "WND", # wind
    "SUN", # solar
    "WAT", # hydro
    "OIL", # oil
    "NG",  # natural gas
    "COL", # coal
    "NUC", # nuclear
    "OTH", # other
]

# Renewable energy types
rn_list = ["WND", "SUN", "WAT"]

# Carbon intensity of the energy types, gCO2eq/kWh
carbon_intensity = {
    "WND": 11,
    "SUN": 41,
    "WAT": 24,
    "OIL": 650,
    "NG":  490,
    "COL": 820,
    "NUC": 12,
    "OTH": 230,
}

def normalize_to_utc(timestamp):
    """
    Normalize a given timestamp to UTC.
    If the timestamp has no timezone information, it localizes it to UTC.
    """
    if timestamp.tzinfo is None:
        return timestamp.tz_localize('UTC')
    else:
        return timestamp.tz_convert('UTC')


# Construct dataframe from json
# Target specific balancing authority and day
def extractBARange(ba_idx, start_day, end_day): 
    global eba_json
    start_idx = pd.Timestamp('{0}T00Z'.format(start_day), tz='UTC')
    end_idx = pd.Timestamp('{0}T00Z'.format(end_day), tz='UTC')

    idx = pd.date_range(start_day, end_day, freq="H", tz='UTC')
    ba_list = []

    for ng_idx in ng_list:
        series_idx = 'EBA.{0}-ALL.NG.{1}.H'.format(ba_idx, ng_idx)
        this_json = eba_json[eba_json['series_id'] == series_idx].reset_index(drop=True)
        
        if this_json.empty:
            ba_list.append([0] * idx.shape[0])
            continue

        # Normalize start and end dates to UTC
        start_dat = normalize_to_utc(pd.Timestamp(this_json['start'].iloc[0]))
        end_dat = normalize_to_utc(pd.Timestamp(this_json['end'].iloc[0]))

        # Check if start_idx is less than start_dat and end_idx is greater than end_dat
        if start_idx < start_dat:
            print('Indexed start ({0}) precedes {1} dataset range ({2})'.format(start_idx, ng_idx, start_dat))
        if end_idx > end_dat:
            print('Indexed end ({0}) beyond {1} dataset range ({2})'.format(end_idx, ng_idx, end_dat))

        tuple_list = this_json['data'][0]

        tuple_filtered = list(filter(
            lambda x: (
                (normalize_to_utc(pd.Timestamp(x[0])) >= start_idx) and 
                (normalize_to_utc(pd.Timestamp(x[0])) <= end_idx)
            ), 
            tuple_list
        ))

        df = pd.DataFrame(tuple_filtered, columns=['timestamp', 'power'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.sort_values(by=['timestamp'], ascending=True).set_index('timestamp').reindex(index=idx, fill_value=0).reset_index()
        ba_list.append(df['power'].tolist())

    dfa = pd.DataFrame(np.array(ba_list).transpose(), columns=ng_list)
    dfa = dfa.set_index(idx)
    return dfa


# Calculate carbon intensity of the grid (kg CO2/MWh)
# Takes a dataframe of energy generation as input (i.e. output of extractBARange)
# Returns a time series of carbon intensity dataframe
def calculateAVGCarbonIntensity(db):
    tot_carbon = None
    db[db < 0] = 0
    sum_db = db.sum(axis=1)
    for c in carbon_intensity:
        if tot_carbon is None:
            tot_carbon = carbon_intensity[c]*db[c]
        else:
            tot_carbon = tot_carbon + carbon_intensity[c]*db[c]
    tot_carbon = tot_carbon.div(sum_db).to_frame()
    tot_carbon.rename(columns={tot_carbon.columns[0]: "carbon_intensity"}, inplace=True)
    return tot_carbon
