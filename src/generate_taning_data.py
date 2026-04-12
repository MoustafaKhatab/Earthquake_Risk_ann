import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt

# Haversine function for accurate distance in km
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 6371 * 2 * asin(sqrt(a))

def create_dataset(raw_data_path, num_samples=2000):
    df_raw = pd.read_csv(raw_data_path)
    
    # Define Turkey Bounds
    lat_min, lat_max = 36.0, 42.0
    lon_min, lon_max = 26.0, 45.0
    
    dataset = []

    print(f"Generating {num_samples} points across Turkey...")
    for _ in range(num_samples):
        # Generate a random point in Turkey
        t_lat = np.random.uniform(lat_min, lat_max)
        t_lon = np.random.uniform(lon_min, lon_max)
        
        # Filter raw data to events within ~200km first (to speed up calculation)
        nearby_candidates = df_raw[
            (df_raw['latitude'].between(t_lat-2, t_lat+2)) & 
            (df_raw['longitude'].between(t_lon-2, t_lon+2))
        ]
        
        if not nearby_candidates.empty:
            distances = nearby_candidates.apply(
                lambda row: haversine(t_lon, t_lat, row['longitude'], row['latitude']), axis=1
            )
            
            # Features
            dist_min = distances.min()
            count_radius = len(distances[distances <= 50])
            avg_mag = nearby_candidates[distances <= 50]['mag'].mean() if count_radius > 0 else 0
        else:
            dist_min, count_radius, avg_mag = 500, 0, 0

        # LABEL LOGIC from your proposal 
        # High count > 10 and High Mag > 4.5 (Adjustable thresholds)
        label = 1 if (count_radius >= 10 and avg_mag >= 4.5) else 0
        
        dataset.append([t_lat, t_lon, dist_min, count_radius, avg_mag, label])

    columns = ['latitude', 'longitude', 'distance_min', 'count_radius', 'avg_magnitude', 'label']
    training_df = pd.DataFrame(dataset, columns=columns)
    training_df.to_csv("data/turkey_training_set.csv", index=False)
    print("Training set saved: data/turkey_training_set.csv")

if __name__ == "__main__":
    create_dataset("data/raw_seismic_100y.csv")