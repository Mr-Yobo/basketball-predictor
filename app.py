from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# === Mock training for demonstration ===
X_train = np.random.rand(100, 6)
y_train = np.random.randint(0, 2, 100)
scaler = StandardScaler().fit(X_train)
X_scaled = scaler.transform(X_train)
model = RandomForestClassifier().fit(X_scaled, y_train)

# === FastAPI setup ===
app = FastAPI(title="Basketball Match Predictor API")

class MatchData(BaseModel):
    team_a_last10: list[int]
    team_b_last10: list[int]
    h2h_last10: list[int] = []
    team_a_elo: float = 0.5
    team_b_elo: float = 0.5
    home: int = 0

def make_feature_vector(data: MatchData):
    def summarize(lst):
        if len(lst) == 0: return [0, 0]
        return [np.mean(lst), np.std(lst)]
    a_mean, a_std = summarize(data.team_a_last10)
    b_mean, b_std = summarize(data.team_b_last10)
    h_mean, h_std = summarize(data.h2h_last10)
    return np.array([[a_mean, a_std, b_mean, b_std, data.team_a_elo, data.team_b_elo]])

@app.get("/")
def root():
    return {"message": "🏀 Basketball Match Prediction API is running!"}

@app.post("/predict")
def predict_match(data: MatchData):
    X = make_feature_vector(data)
    Xs = scaler.transform(X)
    proba = model.predict_proba(Xs)[0, 1]
    return {
        "team_a_win_probability": float(round(proba, 3)),
        "team_b_win_probability": float(round(1 - proba, 3)),
        "prediction": "Team A Wins" if proba > 0.5 else "Team B Wins"
    }
    from fastapi import FastAPI
import numpy as np
import random

app = FastAPI()

@app.get("/")
def home():
    return {"message": "🏀 Basketball Predictor API is live!"}

@app.post("/predict/")
def predict(team_a: str, team_b: str):
    # Simulate getting the last 10 results — replace with your actual model later
    team_a_recent = np.random.randint(60, 120, 10)
    team_b_recent = np.random.randint(60, 120, 10)
    
    # Simple logic: team with higher recent average wins
    avg_a = np.mean(team_a_recent)
    avg_b = np.mean(team_b_recent)

    if avg_a > avg_b:
        winner = team_a
    elif avg_b > avg_a:
        winner = team_b
    else:
        winner = random.choice([team_a, team_b])

    return {
        "team_a_avg": avg_a,
        "team_b_avg": avg_b,
        "predicted_winner": winner
    }
    
