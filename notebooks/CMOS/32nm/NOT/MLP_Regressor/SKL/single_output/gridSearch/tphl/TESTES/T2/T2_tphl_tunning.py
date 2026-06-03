# MLP TPHL Training and tuning

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
import pandas as pd
import sys

import joblib

# Loading train data

X_train = pd.read_csv('/home/eduarda/Repos/jupyter_ic/data/processed/CMOS/32nm/NOT/03_train_val_test_splitting/X_train.csv')
y_train = pd.read_csv('/home/eduarda/Repos/jupyter_ic/data/processed/CMOS/32nm/NOT/03_train_val_test_splitting/y_train.csv')

print(f"Training set loaded: {X_train.shape}")
print(f"Validation set loaded: {y_train.shape}")

save_models_dir = '/home/eduarda/Repos/jupyter_ic/models/trained/MLP_Regressor/CMOS/32nm/NOT/single_output/grid_search_tuning_test/tphl/TESTES/T2'

save_log_dir = '/home/eduarda/Repos/jupyter_ic/notebooks/CMOS/32nm/NOT/MLP_Regressor/SKL/single_output/gridSearch/tphl/TESTES/T2'
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

sys.stdout = Logger(f"{save_log_dir}/training_tunning_output_log.txt")
# -

# Path
untrained_model_path = '/home/eduarda/Repos/jupyter_ic/models/untrained/mlp_regressor_SKL_untrained.joblib'

# Loading Scikit-Learn model
mlp_regressor = joblib.load(untrained_model_path)

### Single-output models **SCIKIT-LEARN**

### Predicting tphl

X = X_train
y = y_train['tphl']

my_pipeline = Pipeline([("scaler", StandardScaler()), ("mlp", mlp_regressor)])

model_with_scaler_y = TransformedTargetRegressor(
    regressor=my_pipeline, transformer=StandardScaler()
)

param_grid = {
    "regressor__mlp__hidden_layer_sizes": [(50,),(100,),(50,50),(100, 50),],
    "regressor__mlp__activation": ["tanh", "relu"],
    "regressor__mlp__solver": ["adam"], 
    "regressor__mlp__alpha": [0.0001, 0.001],
    "regressor__mlp__learning_rate_init" : [0.01, 0.001]
}

grid_search = GridSearchCV(
    estimator=model_with_scaler_y, 
    param_grid=param_grid, 
    cv=10, 
    n_jobs=-1, 
    verbose=3
)

grid_search.fit(X, y)

best_parameters = grid_search.best_params_
print(f"Best parameters: {best_parameters}")

best_score = grid_search.best_score_
print(f"Best score (R²): {best_score}")

trained_model = grid_search.best_estimator_
joblib.dump(trained_model, f'{save_models_dir}/best_GS_tphl.joblib')

history = pd.DataFrame(grid_search.cv_results_)
history = history.sort_values(by="rank_test_score")
print(
    history[
        [
            "params",
            "mean_test_score",
            "std_test_score",
            "mean_fit_time",
        ]
    ].head()
)

print("Trained model saved!")

sys.stdout = sys.stdout.terminal

# Comando para rodar:
#time /home/eduarda/Repos/jupyter_ic/venv/bin/python /home/eduarda/Repos/jupyter_ic/notebooks/CMOS/32nm/NOT/MLP_Regressor/SKL/single_output/gridSearch/tphl/TESTES/T2/T2_tphl_tunning.py