from io import StringIO
import os
import pandas as pd
import numpy as np
from itertools import product
from aws_constants import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
try:
    import boto3
except Exception:
    boto3 = None
try:
    from prophet import Prophet
    from prophet.plot import plot_plotly, plot_components_plotly
except Exception:
    Prophet = None
    plot_plotly = None
    plot_components_plotly = None


def get_data(access_key_id: str = None, secret_access_key: str = None):
    local_csv_path = '/workspace/data/GlobalWeatherRepository.csv'
    if access_key_id and secret_access_key and boto3 is not None:
        try:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key
            )
            response = s3_client.get_object(Bucket='weatherforecast-bucket',
                                            Key='GlobalWeatherRepository.csv')
            object_content = response['Body'].read().decode('utf-8')
            weather_df = pd.read_csv(StringIO(object_content))
        except Exception as e:
            print(f"S3 fetch failed, falling back to local CSV. Reason: {e}")
            weather_df = pd.read_csv(local_csv_path)
    else:
        weather_df = pd.read_csv(local_csv_path)

    return weather_df.loc[weather_df.country == 'Belarus']


def transform_data(data):
    forecast_data = data.rename(columns={"last_updated": "ds", "temperature_celsius": "y"})
    forecast_data['ds'] = pd.to_datetime(forecast_data['ds'])
    try:
        forecast_data['ds'] = forecast_data['ds'].dt.tz_localize(None)
    except Exception:
        pass
    forecast_data['ds'] = forecast_data['ds'].dt.date
    forecast_data['ds'] = pd.to_datetime(forecast_data['ds'])
    forecast_data = forecast_data[['ds', 'y']].dropna()

    # Aggregate to daily mean to ensure unique ds
    forecast_data = (
        forecast_data
            .groupby('ds', as_index=False)['y']
            .mean()
            .sort_values('ds')
    )

    # Ensure continuous daily frequency and fill gaps
    full_range = pd.date_range(start=forecast_data['ds'].min(), end=forecast_data['ds'].max(), freq='D')
    forecast_data = (
        forecast_data
            .set_index('ds')
            .reindex(full_range)
            .rename_axis('ds')
            .reset_index()
    )

    # Interpolate missing values
    forecast_data['y'] = forecast_data['y'].interpolate(method='linear', limit_direction='both')
    forecast_data.reset_index(drop=True, inplace=True)
    return forecast_data


def _compute_metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100.0)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def _time_series_train_val_split(df, val_days):
    if len(df) < 14:
        return None, None
    val_days = max(1, min(val_days, len(df) - 1))
    train_df = df.iloc[:-val_days].copy()
    val_df = df.iloc[-val_days:].copy()
    return train_df, val_df


def _has_enough_history(df, min_points: int = 14):
    return len(df) >= min_points


def tune_prophet_hyperparams(forecast_data):
    if Prophet is None or not _has_enough_history(forecast_data):
        return None, None
    val_days = max(14, min(30, max(7, int(len(forecast_data) * 0.2))))
    train_df, val_df = _time_series_train_val_split(forecast_data, val_days)
    if train_df is None or len(train_df) == 0:
        return None, None
    candidate_grid = [
        {
            "seasonality_mode": mode,
            "changepoint_prior_scale": cps,
            "seasonality_prior_scale": sps,
            "changepoint_range": cpr,
            "weekly_seasonality": True,
            "yearly_seasonality": True,
            "daily_seasonality": False,
            "random_state": 42,
        }
        for mode in ["additive", "multiplicative"]
        for cps in [0.05, 0.1, 0.5]
        for sps in [5.0, 10.0]
        for cpr in [0.9]
    ]
    best_params = None
    best_metrics = None
    best_mape = float("inf")
    for params in candidate_grid:
        model = Prophet(**params)
        model.fit(train_df)
        future = model.make_future_dataframe(periods=len(val_df), freq='D')
        preds = model.predict(future)[['ds', 'yhat']]
        merged = pd.merge(val_df[['ds', 'y']], preds, on='ds', how='left')
        metrics = _compute_metrics(merged['y'].values, merged['yhat'].values)
        if metrics["MAPE"] < best_mape:
            best_mape = metrics["MAPE"]
            best_params = params
            best_metrics = metrics
    return best_params, best_metrics


def _tune_sarimax(train_series, val_series):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        exceptions = None
    except Exception as e:
        return None, None
    if len(train_series) < 7 or len(val_series) == 0:
        return None, None
    p_values = [0, 1, 2]
    d_values = [0, 1]
    q_values = [0, 1, 2]
    P_values = [0, 1]
    D_values = [0, 1]
    Q_values = [0, 1]
    m_values = [7]
    best_cfg = None
    best_metrics = None
    best_mape = float("inf")
    for p in p_values:
        for d in d_values:
            for q in q_values:
                for P in P_values:
                    for D in D_values:
                        for Q in Q_values:
                            for m in m_values:
                                try:
                                    model = SARIMAX(train_series, order=(p, d, q), seasonal_order=(P, D, Q, m), enforce_stationarity=False, enforce_invertibility=False)
                                    res = model.fit(disp=False)
                                    pred = res.forecast(steps=len(val_series))
                                    metrics = _compute_metrics(val_series.values, pred.values)
                                    if metrics["MAPE"] < best_mape:
                                        best_mape = metrics["MAPE"]
                                        best_cfg = {"order": (p, d, q), "seasonal_order": (P, D, Q, m)}
                                        best_metrics = metrics
                                except Exception:
                                    continue
    return best_cfg, best_metrics


def _forecast_sarimax(full_series, steps, best_cfg):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    model = SARIMAX(full_series, order=best_cfg["order"], seasonal_order=best_cfg["seasonal_order"], enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    pred = res.get_forecast(steps=steps)
    pred_mean = pred.predicted_mean
    pred_index = pd.date_range(start=full_series.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
    out = pd.DataFrame({"ds": pred_index, "yhat": pred_mean})
    return out


def _seasonal_naive_forecast(series: pd.Series, steps: int, season_length: int = 7):
    last_value = series.iloc[-1]
    if len(series) >= season_length:
        pattern = series.iloc[-season_length:].values
        repeats = int(np.ceil(steps / season_length))
        forecast_values = np.tile(pattern, repeats)[:steps]
    else:
        forecast_values = np.repeat(last_value, steps)
    future_index = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
    return pd.DataFrame({"ds": future_index, "yhat": forecast_values})


def make_forecast(forecast_data, best_params, periods: int = 30):
    if Prophet is not None and best_params is not None and _has_enough_history(forecast_data):
        model = Prophet(**best_params)
        model.fit(forecast_data)
        future = model.make_future_dataframe(periods=periods, freq='D')
        predictions = model.predict(future)
        try:
            plot_plotly(model, predictions)
        except Exception as e:
            print(f"Plotting failed or not supported in this environment: {e}")
        return predictions
    # Try SARIMAX
    val_days = max(14, min(30, max(7, int(len(forecast_data) * 0.2))))
    split = _time_series_train_val_split(forecast_data, val_days)
    if split is not None:
        train_df, val_df = split
    else:
        train_df, val_df = None, None
    best_cfg, val_metrics = (None, None)
    if train_df is not None and val_df is not None:
        best_cfg, val_metrics = _tune_sarimax(train_df.set_index('ds')['y'], val_df.set_index('ds')['y'])
    if best_cfg is not None:
        print(f"[SARIMAX] Validation metrics: {val_metrics}")
        pred_df = _forecast_sarimax(forecast_data.set_index('ds')['y'], periods, best_cfg)
        predictions = forecast_data.merge(pred_df, on='ds', how='outer')
        return predictions
    # Seasonal naive fallback
    print("Using seasonal naive fallback forecast (no Prophet/statsmodels or insufficient data).")
    series = forecast_data.set_index('ds')['y']
    pred_df = _seasonal_naive_forecast(series, periods, season_length=7)
    predictions = forecast_data.merge(pred_df, on='ds', how='outer')
    return predictions


if __name__ == "__main__":
    belarus_weather_df = get_data(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    transformed_data = transform_data(belarus_weather_df)
    best_params, val_metrics = tune_prophet_hyperparams(transformed_data)
    print(f"Best hyperparameters: {best_params}")
    print(f"Validation metrics: {val_metrics}")
    predictions_forecast = make_forecast(transformed_data, best_params, periods=30)
    print(predictions_forecast.tail(10))
