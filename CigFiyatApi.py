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
from sklearn.linear_model import Ridge

app = FastAPI()

# PostgreSQL connection details
DB_PARAMS = {
    "dbname": "cigsutarztalepdb",
    "user": "cigsutarztalep",
    "password": "C!g$ut@rz.2025**",
    "host": "/var/run/postgresql"
}

@app.post("/forecast_fiyat")
async def forecast_price(request: Request):
    payload = await request.json()
    df = pd.DataFrame(payload["data"])
    periods = int(payload.get("periods", 8))  # default 8 quarters

    df = df[df["Date"].notna()]
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.drop(columns=["Aylık Süt Miktarı Akgıda(Ton)"], inplace=True, errors="ignore")

    interpolate_cols = [
        "Kaba Yem Miktarı (Ton)", "Kırmızı et üretim miktarı",
        "İthal Hayvan Sayısı", "Kaba Yem Fiyatı", "sağılan hayvan Sayısı"
    ]
    available_cols = [col for col in interpolate_cols if col in df.columns]
    df[available_cols] = df[available_cols].interpolate(method="time").bfill().ffill()

    df = df.select_dtypes(include=[np.number])
    df = df.rename(columns={"Çiğ Süt Tavsiye Fiyatı  (TL/Litre)": "milk_price"})

    df_q = df.resample("Q").mean()
    forecast_index = pd.date_range(start=df_q.index[-1] + pd.offsets.QuarterEnd(), periods=periods, freq='Q')

    # SARIMAX
    sarimax_model = SARIMAX(df_q["milk_price"], order=(1,1,1), seasonal_order=(1,1,1,4))
    sarimax_result = sarimax_model.fit(disp=False)
    sarimax_forecast = sarimax_result.predict(start=forecast_index[0], end=forecast_index[-1])

    # Prophet
    df_prophet = df_q.reset_index().rename(columns={"Date": "ds", "milk_price": "y"})
    model_prophet = Prophet()
    model_prophet.fit(df_prophet)
    future = model_prophet.make_future_dataframe(periods=periods, freq='Q')
    forecast_prophet = model_prophet.predict(future)
    prophet_forecast = forecast_prophet.set_index("ds").loc[forecast_index]["yhat"]

    # LSTM
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_q)
    target_idx = df_q.columns.get_loc("milk_price")

    def create_seq(data, seq_len=8):
        x, y = [], []
        for i in range(seq_len, len(data)):
            x.append(data[i-seq_len:i])
            y.append(data[i][target_idx])
        return np.array(x), np.array(y)

    X_all, y_all = create_seq(scaled_data)
    X_all = X_all.reshape((X_all.shape[0], X_all.shape[1], scaled_data.shape[1]))

    model_lstm = Sequential()
    model_lstm.add(LSTM(50, return_sequences=False, input_shape=(X_all.shape[1], X_all.shape[2])))
    model_lstm.add(Dense(1))
    model_lstm.compile(optimizer='adam', loss='mse')
    model_lstm.fit(X_all, y_all, epochs=50, verbose=0)

    last_seq = scaled_data[-8:].reshape((1, 8, scaled_data.shape[1]))
    lstm_preds = []
    for _ in range(periods):
        pred_scaled = model_lstm.predict(last_seq)[0][0]
        full_pred = np.zeros((scaled_data.shape[1],))
        full_pred[target_idx] = pred_scaled
        pred = scaler.inverse_transform([full_pred])[0][target_idx]
        lstm_preds.append(pred)
        new_seq = last_seq[0][1:].tolist() + [full_pred]
        last_seq = np.array(new_seq).reshape((1, 8, scaled_data.shape[1]))

    lstm_forecast = pd.Series(lstm_preds, index=forecast_index)

    # Ridge Ensemble
    ensemble_X = pd.DataFrame({
        "sarimax": sarimax_forecast,
        "prophet": prophet_forecast,
        "lstm": lstm_forecast
    })
    ridge = Ridge()
    ridge.fit(ensemble_X, ensemble_X.mean(axis=1))
    ensemble_forecast = pd.Series(ridge.predict(ensemble_X), index=forecast_index)

    # return {"forecast_price": ensemble_forecast.to_dict()}
    
    # Convert Timestamp keys to string
    result_dict = {str(k.date()): v for k, v in ensemble_forecast.items()}
    result_id = str(uuid.uuid4())
    
    # Save to PostgreSQL
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO cigsut_schema.forecast_fiyat (id, results) VALUES (%s, %s)",
        (result_id, json.dumps(result_dict))
    )
    conn.commit()
    cur.close()
    conn.close()
    
    return {"result_id": result_id}

@app.get("/get_result_fiyat/{result_id}")
async def get_result(result_id: str):
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute("SELECT results FROM cigsut_schema.forecast_fiyat WHERE id = %s", (result_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if row:
        return row[0]  # results is already a dict (from JSONB column)
    else:
        return {"error": "ID not found"}
