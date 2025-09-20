from fastapi import FastAPI, Request
import pandas as pd
import numpy as np
import uuid
import psycopg2
import json
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

app = FastAPI()

# PostgreSQL connection details
DB_PARAMS = {
    "dbname": "cigsutarztalepdb",
    "user": "cigsutarztalep",
    "password": "C!g$ut@rz.2025**",
    "host": "/var/run/postgresql"
}

@app.post("/forecast_arz")
async def forecast(request: Request):
    payload = await request.json()
    df = pd.DataFrame(payload["data"])
    periods = int(payload.get("periods", 12))  # default 12 months

    df = df[df["Date"].notna()]
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.drop(columns=["Aylık Süt Miktarı Akgıda(Ton)"], inplace=True, errors="ignore")

    # Automatically interpolate all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="time").bfill().ffill()

    df = df.rename(columns={"Aylık Süt Miktarı Ülke(Ton)": "milk_supply"})
    df = df.sort_index()

    forecast_start = df.index[-1] + pd.DateOffset(months=1)
    forecast_index = pd.date_range(start=forecast_start, periods=periods, freq='MS')

    # SARIMAX
    sarimax_model = SARIMAX(df["milk_supply"], order=(1,1,1), seasonal_order=(1,1,1,12))
    sarimax_result = sarimax_model.fit(disp=False)
    sarimax_forecast = sarimax_result.predict(start=forecast_index[0], end=forecast_index[-1])

    # Prophet
    df_prophet = df.reset_index().rename(columns={"Date": "ds", "milk_supply": "y"})
    correlations = df.corr()['milk_supply'].sort_values(ascending=False)
    top_features = correlations[1:8].index.tolist()
    safe_features = [f for f in top_features if f in df.columns]

    model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    for feature in safe_features:
        model_prophet.add_regressor(feature)
    model_prophet.fit(df_prophet)

    future_dates = pd.DataFrame({"ds": forecast_index})
    for feature in safe_features:
        last_values = df[feature].iloc[-periods:].values if len(df) >= periods else df[feature].values[-1]
        if isinstance(last_values, np.ndarray) and len(last_values) == periods:
            future_dates[feature] = last_values
        else:
            future_dates[feature] = [last_values] * periods

    forecast_prophet = model_prophet.predict(future_dates)
    prophet_forecast = forecast_prophet.set_index("ds")["yhat"]

    # LSTM
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    target_column_idx = df.columns.get_loc("milk_supply")

    def create_seq(data, seq_len=24):
        x, y = [], []
        for i in range(seq_len, len(data)):
            x.append(data[i-seq_len:i])
            y.append(data[i][target_column_idx])
        return np.array(x), np.array(y)

    X_full, y_full = create_seq(scaled_data)
    X_full = X_full.reshape((X_full.shape[0], X_full.shape[1], scaled_data.shape[1]))

    model_lstm = Sequential()
    model_lstm.add(LSTM(50, return_sequences=False, input_shape=(X_full.shape[1], X_full.shape[2])))
    model_lstm.add(Dense(1))
    model_lstm.compile(optimizer='adam', loss='mse')
    model_lstm.fit(X_full, y_full, epochs=50, verbose=0)

    last_sequence = scaled_data[-24:].reshape((1, 24, scaled_data.shape[1]))
    lstm_preds = []
    for _ in range(periods):
        pred_scaled = model_lstm.predict(last_sequence)[0][0]
        full_pred = np.zeros((scaled_data.shape[1],))
        full_pred[target_column_idx] = pred_scaled
        pred = scaler.inverse_transform([full_pred])[0][target_column_idx]
        lstm_preds.append(pred)
        new_entry = last_sequence[0][1:].tolist() + [full_pred]
        last_sequence = np.array(new_entry).reshape((1, 24, scaled_data.shape[1]))

    lstm_forecast = pd.Series(lstm_preds, index=forecast_index)

    # XGBoost
    X = df.drop(columns=["milk_supply"])
    y = df["milk_supply"]
    model_xgb = xgb.XGBRegressor(n_estimators=100)
    model_xgb.fit(X, y)

    last_values = X.iloc[-periods:].values if len(X) >= periods else np.tile(X.iloc[-1].values, (periods, 1))
    xgb_forecast = pd.Series(model_xgb.predict(last_values), index=forecast_index)

    # Bagging Ensemble
    ensemble_X = pd.DataFrame({
        "sarimax": sarimax_forecast,
        "prophet": prophet_forecast,
        "lstm": lstm_forecast,
        "xgb": xgb_forecast
    })
    bagging_model = RandomForestRegressor(n_estimators=100, random_state=42)
    bagging_model.fit(ensemble_X, ensemble_X.mean(axis=1))
    bagging_forecast = pd.Series(bagging_model.predict(ensemble_X), index=forecast_index)

    # return {"forecast": bagging_forecast.to_dict()}
    
  # Convert Timestamp keys to string
    result_dict = {str(k.date()): v for k, v in bagging_forecast.items()}
    result_id = str(uuid.uuid4())
    
    # Save to PostgreSQL
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO cigsut_schema.forecast_arz (id, results) VALUES (%s, %s)",
        (result_id, json.dumps(result_dict))
    )
    conn.commit()
    cur.close()
    conn.close()
    
    return {"result_id": result_id}

@app.get("/get_result_arz/{result_id}")
async def get_result(result_id: str):
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute("SELECT results FROM cigsut_schema.forecast_arz WHERE id = %s", (result_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return json.loads(row[0]) if row else {"error": "ID not found"}
  
