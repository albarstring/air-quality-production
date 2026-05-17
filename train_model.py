"""
train_model.py
Jalankan sekali untuk menghasilkan model_output/LightGBM.pkl dan model_output/scaler.pkl
  python train_model.py
"""

import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import warnings, os

warnings.filterwarnings("ignore")
np.random.seed(42)

os.makedirs("model_output", exist_ok=True)

# ── Generate dataset sintetis realistis ────────────────────────────────────────
def generate_dataset(n_days=120, seed=42):
    np.random.seed(seed)
    N = n_days * 24
    start = datetime(2024, 1, 1)
    timestamps = [start + timedelta(hours=i) for i in range(N)]
    hours   = np.array([t.hour for t in timestamps])
    weekday = np.array([t.weekday() for t in timestamps])

    is_weekday   = (weekday < 5).astype(float)
    rush_morning = ((hours >= 7) & (hours <= 9)).astype(float)
    rush_evening = ((hours >= 17) & (hours <= 19)).astype(float)
    traffic = (1.0 + 0.6*rush_morning + 0.5*rush_evening) * np.where(is_weekday, 1.0, 0.65)

    temp     = 27 + 5*np.sin(2*np.pi*hours/24 - np.pi/2) + np.random.normal(0, 1.5, N)
    humidity = np.clip(78 - 0.7*temp + np.random.normal(0, 5, N), 30, 99)
    wind     = np.abs(2.5 + 1.5*np.sin(2*np.pi*hours/24) + np.random.normal(0, 0.5, N))
    rain     = np.random.binomial(1, 0.15, N) * np.abs(np.random.normal(3, 2, N))
    rain_eff = np.clip(1 - rain*0.05, 0.6, 1.0)
    wind_eff = 1 / (1 + 0.15*wind)

    pm25 = np.clip((35 + 40*traffic)*wind_eff*rain_eff + np.random.normal(0, 8, N), 5, 350)
    pm10 = np.clip(pm25*1.45 + np.random.normal(0, 10, N), 10, 450)
    co   = np.clip(0.4 + 0.025*pm25 + np.random.normal(0, 0.2, N), 0.1, 10)
    no2  = np.clip(15 + 0.35*pm25*traffic + np.random.normal(0, 8, N), 2, 200)
    o3   = np.clip(40 + 5*(temp-25) - 0.1*no2 + np.random.normal(0, 5, N), 0, 180)

    def pm25_to_aqi(c):
        bp = [(0,12,0,50),(12,35.4,51,100),(35.4,55.4,101,150),
              (55.4,150.4,151,200),(150.4,250.4,201,300)]
        aqi = np.zeros_like(c)
        for lo_c,hi_c,lo_a,hi_a in bp:
            mask = (c >= lo_c) & (c <= hi_c)
            aqi[mask] = lo_a + (c[mask]-lo_c)/(hi_c-lo_c)*(hi_a-lo_a)
        aqi[c > 250.4] = 301
        return aqi

    aqi = np.clip(pm25_to_aqi(pm25) + np.random.normal(0, 4, N), 0, 350)

    return pd.DataFrame({
        "timestamp": timestamps, "hour": hours, "weekday": weekday,
        "is_weekend": (~is_weekday.astype(bool)).astype(int),
        "pm25": pm25.round(2), "pm10": pm10.round(2), "co": co.round(3),
        "no2": no2.round(2), "o3": o3.round(2), "temperature": temp.round(1),
        "humidity": humidity.round(1), "wind_speed": wind.round(2),
        "precipitation": rain.round(2), "traffic_factor": traffic.round(2),
        "aqi": aqi.round(1),
    })

df = generate_dataset(n_days=120)

# ── Feature engineering ────────────────────────────────────────────────────────
df["pm25_pm10_ratio"]        = df["pm25"] / (df["pm10"] + 1e-6)
df["heat_index"]             = df["temperature"] + 0.33*(df["humidity"]/100)*6.105*np.exp(17.27*df["temperature"]/(237.7+df["temperature"]))-4
df["is_rush_hour"]           = ((df["hour"].between(7,9))|(df["hour"].between(17,19))).astype(int)
df["is_daytime"]             = df["hour"].between(6,18).astype(int)
df["sin_hour"]               = np.sin(2*np.pi*df["hour"]/24)
df["cos_hour"]               = np.cos(2*np.pi*df["hour"]/24)
df["sin_weekday"]            = np.sin(2*np.pi*df["weekday"]/7)
df["total_polutan"]          = df["pm25"] + df["pm10"] + df["no2"]*0.1 + df["co"]*10
df["wind_humidity_interact"] = df["wind_speed"] * df["humidity"]

FEATURES = [
    "pm25","pm10","co","no2","o3",
    "temperature","humidity","wind_speed","precipitation",
    "pm25_pm10_ratio","heat_index","total_polutan",
    "is_rush_hour","is_daytime","is_weekend",
    "sin_hour","cos_hour","sin_weekday",
    "wind_humidity_interact",
]

X = df[FEATURES]
y = df["aqi"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Train LightGBM ─────────────────────────────────────────────────────────────
model = lgb.LGBMRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.03,
    subsample=0.85, colsample_bytree=0.85,
    reg_alpha=0.1, reg_lambda=1.0,
    min_child_samples=10, num_leaves=63,
    random_state=42, verbose=-1,
)
model.fit(X_train_s, y_train,
          eval_set=[(X_test_s, y_test)],
          callbacks=[lgb.early_stopping(50, verbose=False),
                     lgb.log_evaluation(period=-1)])

pred  = model.predict(X_test_s)
mae   = mean_absolute_error(y_test, pred)
rmse  = mean_squared_error(y_test, pred)**0.5
r2    = r2_score(y_test, pred)

print(f"MAE  = {mae:.3f}")
print(f"RMSE = {rmse:.3f}")
print(f"R²   = {r2:.4f}")

joblib.dump(model,  "model_output/LightGBM.pkl")
joblib.dump(scaler, "model_output/scaler.pkl")

meta = {
    "best_model": "LightGBM",
    "features": FEATURES,
    "metrics": {"MAE": round(mae,3), "RMSE": round(rmse,3), "R2": round(r2,4)},
    "trained_at": datetime.now().isoformat(),
    "n_train": int(X_train.shape[0]),
    "n_test":  int(X_test.shape[0]),
}
with open("model_output/metadata.json","w") as f:
    json.dump(meta, f, indent=2)

print("✅ Model, scaler, dan metadata tersimpan di model_output/")
