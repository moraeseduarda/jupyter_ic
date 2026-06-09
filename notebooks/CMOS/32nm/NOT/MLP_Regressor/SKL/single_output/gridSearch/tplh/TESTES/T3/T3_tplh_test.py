import sys
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import joblib

X_train = pd.read_csv('/home/eduarda/Repos/jupyter_ic/data/processed/CMOS/32nm/NOT/03_train_val_test_splitting/X_train.csv')
y_train = pd.read_csv('/home/eduarda/Repos/jupyter_ic/data/processed/CMOS/32nm/NOT/03_train_val_test_splitting/y_train.csv')

save_log_dir = '/home/eduarda/Repos/jupyter_ic/notebooks/CMOS/32nm/NOT/MLP_Regressor/SKL/single_output/gridSearch/tplh/TESTES/T3'

# -------------------------------------------------------------------------
# LOG SAVING
# -------------------------------------------------------------------------
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(f"{save_log_dir}/testing_output_log.txt")
# -

# Loading models
tplh_model = joblib.load('/home/eduarda/Repos/jupyter_ic/models/trained/MLP_Regressor/CMOS/32nm/NOT/single_output/grid_search_tuning_test/tplh/TESTES/T3/best_GS_tplh.joblib')

X = X_train
y = y_train['tplh']


model_scaler_y = tplh_model.transformer_

# Loading split data
X_test = pd.read_csv('/home/eduarda/Repos/jupyter_ic/data/processed/CMOS/32nm/NOT/03_train_val_test_splitting/X_test.csv')
y_test = pd.read_csv('/home/eduarda/Repos/jupyter_ic/data/processed/CMOS/32nm/NOT/03_train_val_test_splitting/y_test.csv')

print(f"X Test set loaded: {X_test.shape}")
print(f"Y Test set loaded: {y_test.shape}")
print(f"Y tplh Test set loaded: {y_test['tplh'].shape}")

# Testing
print("\nTesting tplh")
y_pred = tplh_model.predict(X_test)

# Testing R² score
r2 = r2_score(y_test['tplh'], y_pred)
print(f"R²: {r2}")

# Testing error
rmse_physical = root_mean_squared_error(y_test['tplh'], y_pred)
rmse_ps = rmse_physical * 1e12

mape = mean_absolute_percentage_error(y_test['tplh'], y_pred)

print(f'MLP, RMSE physical: {rmse_physical}')
print(f'MLP, RMSE physical (ps): {rmse_ps:.6f} ps')
print(f'MLP, MAPE: {mape * 100:.4f}%')


residuals = y_test['tplh'] - y_pred

plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.1, color='purple')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot for tplh MLP Model (Physical Scale)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals / Errors')
plt.grid(True, alpha=0.3)
plt.savefig("/home/eduarda/Repos/jupyter_ic/notebooks/CMOS/32nm/NOT/MLP_Regressor/SKL/single_output/gridSearch/tplh/TESTES/T3/residuals_physical"
"/residuals.png")
plt.show()
plt.close()

sys.stdout = sys.stdout.terminal