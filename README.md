# Earthquake_Risk_ann
This project builds an ANN-based seismic risk classifier. It analyzes historical earthquake data to label locations as Safe or Risky by learning patterns from past activity. It doesn’t predict future earthquakes, but identifies areas with characteristics similar to historically high-risk zones.

# task one was to clean and work with only turkiye data
data/turkey_training_set.csv has all the data for the past 100 yeays ago

# features
    latitude: a geographic coordinate that specifies the north-south position of a point on the surface of the Earth or another celestial body.
    longitude: is a geographic coordinate that specifies the east-west position of a point on the surface of the Earth
    distance_min: How close is the nearest epicenter?
    count_radius: How many earthquakes happened within 50km?
    avg_magnitude: What is the average strength of those nearby events?

# lable 
    (0,1) : it is out lable , 0 means it no effect from this earthquike and 1 means it is effecting or there is a noticed impact

# 