import pandas as pd
from utils import preprocess_data, get_predictions, run_simulations, calculate_custom_hit_ratio, calculate_LRU_hit_ratio, calculate_FIFO_hit_ratio, visualize_results

# Load data
data = pd.read_csv('GBvideos_with_quality.csv')

data, quality_encoder = preprocess_data(data)

category_ids = data['category_id'].unique()

# Collect all forecasts
all_forecasts = []
for category_id in category_ids:
    category_data = data[data['category_id'] == category_id]
    forecast = get_predictions(category_data)
    all_forecasts.append(forecast)

# Combine all forecasts
all_forecasts_df = pd.concat(all_forecasts).reset_index(drop=True)
all_forecasts_df['adjusted_views'] = all_forecasts_df['yhat']  # Assuming adjusted_views is based on yhat

# Define dictionary for algorithms
algorithms = {
    'Custom': calculate_custom_hit_ratio,
    'LRU': calculate_LRU_hit_ratio,
    'FIFO': calculate_FIFO_hit_ratio
}

# Initialize result storage
results = {
    'video_counts': [],
    'user_counts': [],
    'server_counts': [],
    'server_capacities': [],
    'quality_levels': []
}

# Use algorithms
video_counts = [100, 500, 1000, 1500, 2000]
for count in video_counts:
    subset = all_forecasts_df.sample(n=count)
    for alg_name, alg_func in algorithms.items():
        cache_hit_ratio = run_simulations(alg_func, subset, cache_size=500)
        results['video_counts'].append((count, 'All Qualities', alg_name, cache_hit_ratio))

# Simulate different user counts
user_counts = [1000, 5000, 10000, 20000, 50000]
for users in user_counts:
    subset = all_forecasts_df.sample(n=1000)
    for alg_name, alg_func in algorithms.items():
        cache_hit_ratio = run_simulations(alg_func, subset, cache_size=500)
        results['user_counts'].append((users, 'All Qualities', alg_name, cache_hit_ratio))

# Simulate different server counts
server_counts = [1, 5, 10, 20, 50]
for servers in server_counts:
    subset = all_forecasts_df.sample(n=1000)
    for alg_name, alg_func in algorithms.items():
        cache_hit_ratio = run_simulations(alg_func, subset, cache_size=500)
        results['server_counts'].append((servers, 'All Qualities', alg_name, cache_hit_ratio))

# Simulate different server capacities
server_capacities = [100, 300, 500, 700, 1000]
for capacity in server_capacities:
    subset = all_forecasts_df
    for alg_name, alg_func in algorithms.items():
        cache_hit_ratio = run_simulations(alg_func, subset, cache_size=capacity)
        results['server_capacities'].append((capacity, 'All Qualities', alg_name, cache_hit_ratio))

# Simulate different video qualities
quality_levels = data['quality_original'].unique()
for quality in quality_levels:
    subset = all_forecasts_df[all_forecasts_df['quality'] == quality_encoder.transform([quality])[0]]
    if not subset.empty:
        for alg_name, alg_func in algorithms.items():
            cache_hit_ratio = run_simulations(alg_func, subset, cache_size=500)
            results['quality_levels'].append((quality, alg_name, cache_hit_ratio))
    else:
        print(f"No data available for quality: {quality}, skipping.")

# Visualizing the results
visualize_results(results)