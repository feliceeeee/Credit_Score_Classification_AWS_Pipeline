from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class ModelEvaluator:

    def run(self, model_paths, x_test, y_test):
        results = []

        for model_name, model_path in model_paths.items():
            model = joblib.load(Path(model_path))

            y_pred = model.predict(x_test)
            y_proba = model.predict_proba(x_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            weighted_f1 = f1_score(y_test, y_pred, average="weighted")
            test_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

            results.append({"Model": model_name, "Accuracy": accuracy, "Weighted F1": weighted_f1, "Test AUC": test_auc})

        return results