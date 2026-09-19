from pathlib import Path
import shutil
import joblib

import pandas as pd

from data_ingestion import DataIngestion
from preprocessing import CreditScorePreprocessor
from train import CreditScoreTrainer
from evaluation import ModelEvaluator


class CreditScorePipeline:
    def __init__(self, raw_data_path, accuracy_threshold=0.72):
        self.base_dir = Path(__file__).parent
        self.raw_data_path = Path(raw_data_path)

        self.ingested_dir = self.base_dir / "ingested"
        self.train_dir = self.base_dir / "train"
        self.test_dir = self.base_dir / "test"
        self.model_dir = self.base_dir / "model"

        self.accuracy_threshold = accuracy_threshold

        self.ingestor = DataIngestion(self.raw_data_path, self.ingested_dir)
        self.preprocessor = CreditScorePreprocessor()
        self.trainer = CreditScoreTrainer()
        self.evaluator = ModelEvaluator()

    def execute(self):
        print(" CREDIT SCORE TRAINING PIPELINE ")
        ingested_file = self.ingestor.run()
        df = self.preprocessor.clean_data(ingested_file)
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(df)

        self.train_dir.mkdir(exist_ok=True)
        self.test_dir.mkdir(exist_ok=True)

        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        train_path = self.train_dir / "train.csv"
        test_path = self.test_dir / "test.csv"

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"Train saved -> {train_path}")
        print(f"Test saved  -> {test_path}")

        transformer = self.preprocessor.get_transformer(X_train)

        rf_run = self.trainer.train_random_forest(X_train, y_train, transformer)
        xgb_run = self.trainer.train_xgboost(X_train, y_train, transformer)

        run_ids = {"Random Forest": rf_run, "XGBoost": xgb_run}
        results = self.evaluator.run(run_ids, X_test, y_test)

        print("\nEvaluation Result")
        for r in results:
            print("-" * 50)
            print(f"Model : {r['Model']}")
            print(f"Accuracy : {r['Accuracy']:.4f}")
            print(f"Weighted F1 : {r['Weighted F1']:.4f}")
            print(f"AUC : {r['Test AUC']:.4f}")

        best_model = max(results, key=lambda x: x["Weighted F1"])

        print("\nBest Model")
        print(best_model)
        if best_model["Model"] == "Random Forest":
            source_model = (self.trainer.artifact_dir / "random_forest_pipeline.pkl")
        else:
            source_model = (self.trainer.artifact_dir / "xgboost_pipeline.pkl")

        deployment_model = (self.trainer.artifact_dir / "best_model.pkl")

        shutil.copy(source_model, deployment_model)

        print(f"Best model copied to {deployment_model}")


        self.model_dir.mkdir(exist_ok=True)
        sagemaker_model = self.model_dir / "model_credit.joblib"
        shutil.copy(source_model,sagemaker_model)

        print(f"SageMaker model saved -> {sagemaker_model}")

        if best_model["Accuracy"] >= self.accuracy_threshold:
            print("\nSTATUS : APPROVED")
        else:
            print("\nSTATUS : REJECTED")

if __name__ == "__main__":
    DATASET = Path(__file__).parent / "credit_score.csv"
    pipeline = CreditScorePipeline(raw_data_path=DATASET, accuracy_threshold=0.72)
    pipeline.execute()