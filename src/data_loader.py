import pandas as pd
import requests
import time
from io import StringIO
import os

def download_century_data():
    # Ensure the data directory exists
    os.makedirs('data', exist_ok=True)
    
    all_chunks = []
    # Loop from 1926 to 2026 in 10-year blocks
    for start_year in range(1926, 2026, 10):
        end_year = start_year + 10
        url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
        
        # We use minmagnitude 4.0 to focus on "Risky" events 
        # and keep the file size manageable for your models.
        params = {
            "format": "csv",
            "starttime": f"{start_year}-01-01",
            "endtime": f"{end_year}-01-01",
            "minmagnitude": 4.0, 
            "orderby": "time-asc"
        }
        
        print(f"Downloading block: {start_year} to {end_year}...")
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                df_chunk = pd.read_csv(StringIO(response.text))
                all_chunks.append(df_chunk)
                print(f"   Done! Found {len(df_chunk)} events.")
            else:
                print(f"   Failed block {start_year}. Status: {response.status_code}")
        except Exception as e:
            print(f"   Connection error: {e}")
            
        # Be polite to the USGS server
        time.sleep(1)

    if all_chunks:
        final_df = pd.concat(all_chunks, ignore_index=True)
        # Drop duplicates just in case
        final_df = final_df.drop_duplicates(subset=['time', 'latitude', 'longitude'])
        
        output_path = "data/raw_seismic_100y.csv"
        final_df.to_csv(output_path, index=False)
        print(f"\nSUCCESS: Total {len(final_df)} events saved to {output_path}")
    else:
        print("No data was downloaded.")

if __name__ == "__main__":
    download_century_data()