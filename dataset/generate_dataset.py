"""
Dataset Generator for Smart Irrigation Prediction System
Generates realistic agricultural and agro-meteorological data based on
soil physics and crop evapotranspiration (FAO-56 Penman-Monteith principles).

Features:
- Soil Moisture (%) [Range: 5% - 85%]
- Temperature (°C) [Range: 10°C - 48°C]
- Humidity (%) [Range: 15% - 98%]
- Rainfall (mm) [Range: 0mm - 120mm]

Target:
- Irrigation_Required (1 = Yes, 0 = No)
"""

import os
import numpy as np
import pandas as pd

def generate_irrigation_dataset(n_samples=1200, random_state=42):
    np.random.seed(random_state)

    # 1. Soil Moisture (%): normal distribution centered at 35% with realistic spread
    soil_moisture = np.random.uniform(8.0, 80.0, n_samples)

    # 2. Temperature (°C): temperate to tropical climate range
    temperature = np.random.uniform(14.0, 46.0, n_samples)

    # 3. Relative Humidity (%): 15% (arid) to 95% (tropical monsoon)
    # Typically inversely related to temperature with some noise
    humidity = np.clip(105.0 - (temperature * 1.5) + np.random.normal(0, 12, n_samples), 15.0, 95.0)

    # 4. Rainfall (mm): zero-inflated distribution (many dry days, occasional rain)
    rain_probability = np.random.rand(n_samples)
    rainfall = np.where(rain_probability > 0.65, np.random.exponential(scale=18.0, size=n_samples), 0.0)
    rainfall = np.clip(rainfall, 0.0, 110.0)

    # Agronomic Decision Logic for Irrigation (Penman-Monteith inspired):
    # - Soil moisture below permanent wilting point / depletion threshold (~28-32%) strongly indicates irrigation.
    # - High temperature and low humidity accelerate evapotranspiration (ET0).
    # - Rainfall directly suppresses the need for irrigation.

    irrigation_required = []

    for sm, temp, hum, rain in zip(soil_moisture, temperature, humidity, rainfall):
        # Evaporative demand factor
        et_factor = (temp / 30.0) * ((100.0 - hum) / 50.0)
        
        # Effective moisture availability index
        # Rainfall replenishes soil moisture temporarily
        effective_moisture = sm + (rain * 1.8)

        # Baseline threshold influenced by atmospheric demand
        moisture_threshold = 28.0 + (et_factor * 4.5)

        # Immediate override: if heavy rain occurred (> 12 mm), no irrigation needed
        if rain >= 12.0:
            decision = 0
        elif effective_moisture < moisture_threshold:
            # If effective moisture is lower than demand threshold -> Irrigation needed
            # Add minor sensor noise / soil heterogeneity (5% stochastic boundary uncertainty)
            noise = np.random.normal(0, 1.5)
            decision = 1 if (effective_moisture + noise) < moisture_threshold else 0
        else:
            decision = 0

        irrigation_required.append(int(decision))

    df = pd.DataFrame({
        'soil_moisture': np.round(soil_moisture, 2),
        'temperature': np.round(temperature, 2),
        'humidity': np.round(humidity, 2),
        'rainfall': np.round(rainfall, 2),
        'irrigation_required': irrigation_required
    })

    return df

if __name__ == '__main__':
    dataset_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(dataset_dir, 'irrigation_data.csv')
    df = generate_irrigation_dataset(n_samples=1200, random_state=42)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created at: {output_path}")
    print(f"Total Records: {len(df)}")
    print(f"Class Balance:\n{df['irrigation_required'].value_counts(normalize=True)}")
