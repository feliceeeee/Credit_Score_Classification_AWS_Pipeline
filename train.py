from pathlib import Path
import shutil
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
    
class CreditScoreTrainer:

    def __init__(self, artifact_path="artifacts", random_state=42):
        self.random_state = random_state

        self.artifact_dir = Path(artifact_path)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def train_random_forest(self, x_train, y_train, transformer):
        rf_pipeline = Pipeline([('preprocessing', transformer),
                                ('model', RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=5, class_weight='balanced', random_state=self.random_state))])
        
        rf_pipeline.fit(x_train, y_train)
        model_path = (self.artifact_dir / "random_forest_pipeline.pkl")
        joblib.dump(rf_pipeline, model_path)
        print(f"Random Forest saved to {model_path}")
        return model_path
        
    def train_xgboost(self, x_train, y_train, transformer):
        xgb_pipeline = Pipeline([('preprocessing', transformer),
                                 ('model', XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss', learning_rate=0.1, max_depth=6, n_estimators=300, random_state=self.random_state))])

        xgb_pipeline.fit(x_train, y_train)
        model_path = (self.artifact_dir / "xgboost_pipeline.pkl")
        joblib.dump(xgb_pipeline, model_path)
        print(f"XGBoost saved to {model_path}")
        return model_path

    def save_best_model(self, best_model_name, output_dir="model", output_name="model_credit.joblib",):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if best_model_name == "Random Forest":
            source = self.artifact_dir / "random_forest_pipeline.pkl"
        elif best_model_name == "XGBoost":
            source = self.artifact_dir / "xgboost_pipeline.pkl"
        else:
            raise ValueError(f"Unknown model : {best_model_name}")

        destination = output_dir / output_name
        shutil.copy(source, destination)

        print("Best model exported successfully")
        print(f"Source : {source}")
        print(f"Destination : {destination}")

        return destination

    def load_pipeline(self, model_name):
        if model_name == "Random Forest":
            return joblib.load(self.artifact_dir / "random_forest_pipeline.pkl")
        if model_name == "XGBoost":
            return joblib.load(self.artifact_dir / "xgboost_pipeline.pkl")

        raise ValueError(f"Unknown model : {model_name}")