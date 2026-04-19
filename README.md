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

# how we gonna use this data?
 - we use the lat and lon to know where the earthquike happend.
 - we used this points also to point the country we wanna work with 

# our Perceptron equation 
 - The Perceptron takes your features, multiplies them by Weights ($W$), adds a Bias ($b$), and then checks if the total is greater than zero:
 - z = (w1 . lat) + (w2 . lon) + (w3 . dist_min) + (w4 count) + (w5 . avg_mag) + b

 * Activation: If z = 0, output is 1 (Risky). Otherwise, output is 0 (Safe).