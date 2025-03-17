import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def preprocess_data(data):
    # Convert date format
    data['publish_time'] = pd.to_datetime(data['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')

    # Handle missing values
    data = data.dropna(subset=['views', 'likes', 'publish_time', 'tags', 'category_id', 'quality', 'adjusted_views', 'comment_count'])

    # Preserve the original quality field before encoding
    data['quality_original'] = data['quality'].copy()

    # Label encoding for tags
    label_encoder = LabelEncoder()
    data['tags'] = label_encoder.fit_transform(data['tags'])

    # Label encoding for quality
    quality_encoder = LabelEncoder()
    data['quality'] = quality_encoder.fit_transform(data['quality'])

    # Time of day for publishing time
    data['time_of_day'] = data['publish_time'].dt.hour // 6

    return data, quality_encoder

class CustomProphet(Prophet):
    def __init__(self, custom_trend_param=0.5, custom_seasonality_param=1.0, **kwargs):
        super().__init__(**kwargs)
        self.custom_trend_param = custom_trend_param
        self.custom_seasonality_param = custom_seasonality_param
        self.custom_regressors = []

    def add_custom_regressor(self, name, **kwargs):
        self.custom_regressors.append(name)
        self.add_regressor(name, **kwargs)

    def fit(self, df, **kwargs):
        self.add_seasonality(name='quarterly', period=90.25, fourier_order=8)
        super().fit(df, **kwargs)
        self.params['custom_trend'] = self._fit_custom_trend(df)
        self.params['custom_seasonality'] = self._fit_custom_seasonality(df)
        self.params['custom_regressors'] = self._fit_custom_regressors(df)
        return self

    def _fit_custom_trend(self, df):
        dates = pd.to_datetime(df['ds']).map(pd.Timestamp.toordinal).values
        y = df['y'].values
        model = LinearRegression()
        model.fit(dates.reshape(-1, 1), y)
        self.custom_trend_model = model
        return model

    def _predict_custom_trend(self, df):
        if hasattr(self, 'custom_trend_model'):
            dates = pd.to_datetime(df['ds']).map(pd.Timestamp.toordinal).values
            trend = self.custom_trend_model.predict(dates.reshape(-1, 1))
            return trend
        else:
            raise ValueError("Custom trend model has not been fitted.")

    def _fit_custom_seasonality(self, df):
        dates = pd.to_datetime(df['ds']).map(pd.Timestamp.toordinal).values
        y = df['y'].values
        poly = PolynomialFeatures(degree=4)
        dates_poly = poly.fit_transform(dates.reshape(-1, 1))
        model = LinearRegression()
        model.fit(dates_poly, y)
        self.custom_seasonality_model = model
        self.poly = poly
        return model

    def _predict_custom_seasonality(self, df):
        if hasattr(self, 'custom_seasonality_model'):
            dates = pd.to_datetime(df['ds']).map(pd.Timestamp.toordinal).values
            dates_poly = self.poly.transform(dates.reshape(-1, 1))
            seasonality = self.custom_seasonality_model.predict(dates_poly)
            return seasonality
        else:
            raise ValueError("Custom seasonality model has not been fitted.")

    def _fit_custom_regressors(self, df):
        regressors = df[self.custom_regressors]
        y = df['y'].values
        model = LinearRegression()
        model.fit(regressors, y)
        self.custom_regressors_model = model
        return model

    def _predict_custom_regressors(self, df):
        if hasattr(self, 'custom_regressors_model'):
            regressors = df[self.custom_regressors]
            regressors_pred = self.custom_regressors_model.predict(regressors)
            return regressors_pred
        else:
            raise ValueError("Custom regressors model has not been fitted.")

    def predict(self, df):
        forecast = super().predict(df)
        forecast['custom_trend'] = self._predict_custom_trend(df)
        forecast['custom_seasonality'] = self._predict_custom_seasonality(df)
        forecast['custom_regressors'] = self._predict_custom_regressors(df)
        forecast['yhat'] += (forecast['custom_trend'] +
                             forecast['custom_seasonality'] +
                             forecast['custom_regressors'])
        forecast['yhat'] = forecast['yhat'].apply(lambda x: max(0, x))
        return forecast

def get_predictions(category_data):
    df = pd.DataFrame()
    df['ds'] = category_data['publish_time']
    df['y'] = category_data['adjusted_views']
    df['tags'] = category_data['tags']
    df['likes'] = category_data['likes']
    df['quality'] = category_data['quality']
    df['time_of_day'] = category_data['time_of_day']

    df['like_rate'] = category_data['likes'] / category_data['views']
    df['days_since_publish'] = (df['ds'] - df['ds'].min()).dt.days
    df['is_weekend'] = df['ds'].dt.weekday.isin([5, 6]).astype(int)
    df['comment_count'] = category_data['comment_count']

    model = CustomProphet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, seasonality_mode='multiplicative')
    model.add_custom_regressor('tags')
    model.add_custom_regressor('likes')
    model.add_custom_regressor('quality')
    model.add_custom_regressor('days_since_publish')
    model.add_custom_regressor('is_weekend')
    model.add_custom_regressor('time_of_day')
    model.add_custom_regressor('like_rate')
    model.add_custom_regressor('comment_count')

    model.fit(df)

    future = model.make_future_dataframe(periods=365)
    future['tags'] = 0
    future['likes'] = 0
    future['quality'] = 0
    future['time_of_day'] = 0
    future['days_since_publish'] = (future['ds'] - df['ds'].min()).dt.days
    future['is_weekend'] = future['ds'].dt.weekday.isin([5, 6]).astype(int)
    future['like_rate'] = 0
    future['comment_count'] = 0

    forecast = model.predict(future)
    forecast['category_id'] = category_data['category_id'].iloc[0]
    forecast['quality'] = category_data['quality'].iloc[0]
    return forecast

def run_simulations(alg_func, df, cache_size, num_simulations=5):
    total_hit_ratio = 0
    for _ in range(num_simulations):
        total_hit_ratio += alg_func(df, cache_size)
    avg_hit_ratio = total_hit_ratio / num_simulations
    return avg_hit_ratio

def calculate_custom_hit_ratio(df, cache_size, total_requests=1000):
    df_sorted = df.sort_values(by='yhat', ascending=False)
    cached_videos = df_sorted.head(cache_size)
    cached_video_ids = set(cached_videos.index)

    hits = 0
    for i in range(total_requests):
        probabilities = (df['yhat'] * 2 + df['likes'] * 1.5 + df['comment_count'] * 1.2 + df['adjusted_views']).fillna(0) + 1e-9
        probabilities = probabilities.clip(lower=0)
        if probabilities.sum() > 0:
            probabilities = probabilities / probabilities.sum()
        else:
            probabilities = np.ones(len(probabilities)) / len(probabilities)
        requested_video = np.random.choice(df.index, p=probabilities)
        if requested_video in cached_video_ids:
            hits += 1

    cache_hit_ratio = hits / total_requests
    return cache_hit_ratio

# LRU
def calculate_LRU_hit_ratio(df, cache_size, total_requests=1000):
    df_sorted = df.sort_values(by='ds', ascending=False)
    cached_videos = df_sorted.head(cache_size)
    cached_video_ids = set(cached_videos.index)

    hits = 0
    for i in range(total_requests):
        probabilities = (df['yhat'] * 2 + df['likes'] * 1.5 + df['comment_count'] * 1.2 + df['adjusted_views']).fillna(0) + 1e-9

        probabilities = probabilities.clip(lower=0)
        if probabilities.sum() > 0:
            probabilities = probabilities / probabilities.sum()
        else:
            probabilities = np.ones(len(probabilities)) / len(probabilities)

        requested_video = np.random.choice(df.index, p=probabilities)
        if requested_video in cached_video_ids:
            hits += 1

    cache_hit_ratio = hits / total_requests
    return cache_hit_ratio

# FIFO
def calculate_FIFO_hit_ratio(df, cache_size, total_requests=1000):
    df_sorted = df.sort_values(by='ds', ascending=True)
    cached_videos = df_sorted.head(cache_size)
    cached_video_ids = set(cached_videos.index)

    hits = 0
    for i in range(total_requests):
        probabilities = (df['yhat'] * 2 + df['likes'] * 1.5 + df['comment_count'] * 1.2 + df['adjusted_views']).fillna(0) + 1e-9

        probabilities = probabilities.clip(lower=0)
        if probabilities.sum() > 0:
            probabilities = probabilities / probabilities.sum()
        else:
            probabilities = np.ones(len(probabilities)) / len(probabilities)

        requested_video = np.random.choice(df.index, p=probabilities)
        if requested_video in cached_video_ids:
            hits += 1

    cache_hit_ratio = hits / total_requests
    return cache_hit_ratio

def visualize_results(results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparison of Caching Algorithms')

    # Plot by video count
    video_counts_df = pd.DataFrame(results['video_counts'], columns=['Video Count', 'Quality', 'Algorithm', 'Hit Ratio'])
    for alg in video_counts_df['Algorithm'].unique():
        subset = video_counts_df[video_counts_df['Algorithm'] == alg]
        axes[0, 0].plot(subset['Video Count'], subset['Hit Ratio'], label=alg)
    axes[0, 0].set_title('Hit Ratio by Video Count')
    axes[0, 0].set_xlabel('Video Count')
    axes[0, 0].set_ylabel('Hit Ratio')
    axes[0, 0].legend()

    # Plot by user count
    user_counts_df = pd.DataFrame(results['user_counts'], columns=['User Count', 'Quality', 'Algorithm', 'Hit Ratio'])
    for alg in user_counts_df['Algorithm'].unique():
        subset = user_counts_df[user_counts_df['Algorithm'] == alg]
        axes[0, 1].plot(subset['User Count'], subset['Hit Ratio'], label=alg)
    axes[0, 1].set_title('Hit Ratio by User Count')
    axes[0, 1].set_xlabel('User Count')
    axes[0, 1].set_ylabel('Hit Ratio')
    axes[0, 1].legend()

    # Plot by server count
    server_counts_df = pd.DataFrame(results['server_counts'], columns=['Server Count', 'Quality', 'Algorithm', 'Hit Ratio'])
    for alg in server_counts_df['Algorithm'].unique():
        subset = server_counts_df[server_counts_df['Algorithm'] == alg]
        axes[1, 0].plot(subset['Server Count'], subset['Hit Ratio'], label=alg)
    axes[1, 0].set_title('Hit Ratio by Server Count')
    axes[1, 0].set_xlabel('Server Count')
    axes[1, 0].set_ylabel('Hit Ratio')
    axes[1, 0].legend()

    # Plot by server capacity
    server_capacities_df = pd.DataFrame(results['server_capacities'], columns=['Server Capacity', 'Quality', 'Algorithm', 'Hit Ratio'])
    for alg in server_capacities_df['Algorithm'].unique():
        subset = server_capacities_df[server_capacities_df['Algorithm'] == alg]
        axes[1, 1].plot(subset['Server Capacity'], subset['Hit Ratio'], label=alg)
    axes[1, 1].set_title('Hit Ratio by Server Capacity')
    axes[1, 1].set_xlabel('Server Capacity')
    axes[1, 1].set_ylabel('Hit Ratio')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

    # Plot by video quality as a bar chart
    plt.figure(figsize=(10, 6))
    quality_levels_df = pd.DataFrame(results['quality_levels'], columns=['Quality', 'Algorithm', 'Hit Ratio'])
    if not quality_levels_df.empty:
        quality_levels_df = quality_levels_df.pivot(index='Algorithm', columns='Quality', values='Hit Ratio')
        quality_levels_df.plot(kind='bar', ax=plt.gca())
        plt.title('Hit Ratio by Video Quality')
        plt.xlabel('Algorithm')
        plt.ylabel('Hit Ratio')
        plt.legend(title='Video Quality')
        plt.show()
    else:
        print("No data available for plotting the bar chart.")