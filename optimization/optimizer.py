import optuna
import torch
import numpy as np
from optimization.surrogate_model import DistillationSurrogate
import joblib

class DistillationOptimizer:
    def __init__(self):
        self.model = DistillationSurrogate()
        self.model.load_state_dict(torch.load('models/surrogate_model.pth'))
        self.model.eval()
        self.scaler_X = joblib.load('models/scaler_X.pkl')
        self.scaler_y = joblib.load('models/scaler_y.pkl')
    
    def objective(self, trial, target_purity=0.95):
        reflux_ratio = trial.suggest_float('reflux_ratio', 1.0, 10.0)
        feed_rate = trial.suggest_float('feed_rate', 20.0, 200.0)
        feed_composition = trial.suggest_float('feed_composition', 0.1, 0.9)
        
        X = np.array([[reflux_ratio, feed_rate, feed_composition]])
        X_scaled = self.scaler_X.transform(X)
        X_t = torch.FloatTensor(X_scaled)
        
        with torch.no_grad():
            y_pred = self.model(X_t)
            y_pred = self.scaler_y.inverse_transform(y_pred.numpy())
        
        top_purity, bottom_purity, reboiler_duty, condenser_duty = y_pred[0]
        
        purity_penalty = max(0, target_purity - top_purity) * 10000
        
        return reboiler_duty + purity_penalty
    
    def optimize(self, target_purity=0.95, n_trials=100):
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self.objective(trial, target_purity), n_trials=n_trials)
        
        print(f"\nBest parameters: {study.best_params}")
        print(f"Minimum reboiler duty: {study.best_value:.2f} kW")
        
        return study.best_params, study.best_value