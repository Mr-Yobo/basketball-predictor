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
        "team_a_avg": float(avg_a),
        "team_b_avg": float(avg_b),
        "predicted_winner": winner
    }

