# Credit Score Classification — AWS Pipeline

An end-to-end machine learning project that classifies a customer's credit score into Poor, Standard, or Good categories based on demographic, financial, and credit behavior data. This is the cloud-based counterpart of the local pipeline project: the same OOP-based training pipeline is extended to export a SageMaker-compatible model artifact, deploy it as a real-time SageMaker endpoint, and serve predictions through a Streamlit app hosted on an EC2 instance.

## Highlights

- Exploratory data analysis of customer demographic, financial, and credit behavior data (same analysis as the local pipeline project)
- Data cleaning: handling invalid categorical values, extreme placeholder values, and inconsistent numeric formats stored as text
- Feature engineering: conversion of credit history age into total months, validation of loan count using loan type information, and log transformation of skewed income
- OOP-based training pipeline with separate classes for data ingestion, preprocessing, training, and evaluation
- Baseline modeling and comparison between tuned Random Forest and XGBoost classifiers
- Export of the best model into a SageMaker-compatible artifact (`model_credit.joblib`) packaged as `model.tar.gz`
- Custom SageMaker inference script (`inference.py`) implementing `model_fn`, `input_fn`, `predict_fn`, and `output_fn`
- Model upload to Amazon S3 and deployment as a real-time SageMaker endpoint using the SageMaker Python SDK
- Endpoint testing via direct `sagemaker-runtime` invocation with sample payloads
- Cloud-hosted Streamlit frontend that invokes the SageMaker endpoint via `boto3`
- Automated EC2 provisioning script (`user-data.sh`) that clones the repo, sets up a virtual environment, and runs the Streamlit app as a `systemd` service

## Data

The dataset contains 25,000 records and 29 columns, including:
- Demographic information (age, occupation)
- Financial information (annual income, monthly in-hand salary, outstanding debt, monthly balance)
- Credit account details (number of bank accounts, credit cards, and loans)
- Credit behavior (payment delays, delayed payments, credit inquiries, credit mix, payment behavior)
- Credit history length
- `Credit_Score` as the target variable (Good, Standard, or Poor)

## Model Experiments

**Notebook stage (exploration):** Same EDA, cleaning, and baseline/tuned model comparison as the local pipeline project (Decision Tree, Random Forest, Gradient Boosting, XGBoost), with `class_weight='balanced'` used to address class imbalance.

**Pipeline stage (cloud-ready refactor):** The training pipeline mirrors the local version but is adapted for SageMaker deployment:

- `DataIngestion` — validates and saves the raw dataset
- `CreditScorePreprocessor` — cleans data, engineers features, splits train/test, saves the split CSVs (`train/train.csv`, `test/test.csv`), and builds the `ColumnTransformer` (median imputation for numeric features; ordinal encoding for `Credit_Mix` and `Payment_of_Min_Amount`; one-hot encoding for `Payment_Behaviour`)
- `CreditScoreTrainer` — trains a tuned Random Forest (`n_estimators=500`, `min_samples_split=5`) and a tuned XGBoost (`learning_rate=0.1`, `max_depth=6`, `n_estimators=300`) inside sklearn pipelines, and exports the winning model as `model/model_credit.joblib` via `save_best_model()`
- `ModelEvaluator` — loads each trained model from disk and computes accuracy, weighted F1, and macro test AUC (OvR)
- `CreditScorePipeline` — orchestrates ingestion → preprocessing → training → evaluation, selects the best model by weighted F1, and prepares both the local artifact (`artifacts/best_model.pkl`) and the SageMaker-ready artifact (`model/model_credit.joblib`)

**Deployment stage (AWS):**

- The exported model is compressed into `model.tar.gz` and uploaded to an S3 bucket
- A `SKLearnModel` is deployed to a real-time SageMaker endpoint (`credit-score-endpoint`, instance type `ml.m5.large`), using `inference.py` as the entry point for request handling
- The endpoint is tested directly via `sagemaker-runtime.invoke_endpoint()` with a sample payload
- `app_streamlit.py` provides a web UI that builds the same feature payload and calls the SageMaker endpoint through `boto3`, displaying the predicted class, confidence, and full class probabilities
- `user-data.sh` is an EC2 bootstrap script that installs dependencies, clones the repository, and runs the Streamlit app as a persistent `systemd` service on port 8501

## Results

**Random Forest** was selected as the final model (accuracy 73.1%, exceeding the 0.72 deployment threshold) and exported as the SageMaker-served model. A sample end-to-end invocation of the deployed endpoint returned a `Standard` prediction with class probabilities of `{"Poor": 0.169, "Standard": 0.454, "Good": 0.377}`, confirming the deployed pipeline reproduces the same preprocessing and prediction behavior as the local model.

## Project Structure

* `credit_score.ipynb` — EDA, preprocessing, and initial model experimentation (same as local pipeline)
* `data_ingestion.py` — `DataIngestion` class: validates and saves the raw CSV to `ingested/`
* `preprocessing.py` — `CreditScorePreprocessor` class: cleaning, feature engineering, splitting, transformer construction, and saving train/test splits
* `train.py` — `CreditScoreTrainer` class: trains Random Forest and XGBoost pipelines and exports the best one as a SageMaker-ready artifact
* `evaluation.py` — `ModelEvaluator` class: loads saved models and computes evaluation metrics
* `pipeline.py` — `CreditScorePipeline` class: orchestrates the full ingestion → preprocessing → training → evaluation → export workflow
* `train.ipynb` — notebook version of the pipeline run interactively on a SageMaker notebook instance
* `inference.py` — SageMaker inference handlers (`model_fn`, `input_fn`, `predict_fn`, `output_fn`)
* `deploy_endpoint.ipynb` — packages the model, uploads it to S3, deploys the SageMaker endpoint, tests it, and includes endpoint cleanup utilities
* `app_streamlit.py` — Streamlit frontend that invokes the deployed SageMaker endpoint via `boto3`
* `user-data.sh` — EC2 user-data script for automated deployment of the Streamlit app as a systemd service
* `requirements.txt` — dependencies for the Streamlit app (`streamlit`, `boto3`, `pandas`, `numpy`, `joblib`, `scikit-learn==1.4.2`)

## How to Run

### Notebook (EDA and experimentation)

1. Clone this repository:

```
git clone https://github.com/feliceeeee/Credit_Score_Classification_AWS_Pipeline.git
```

2. Install the required libraries:

```
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost jupyter
```

3. Ensure the dataset is located at: `credit_score.csv`
4. Open and run `credit_score.ipynb`

### Training Pipeline (local run, produces the SageMaker-ready artifact)

```
pip install scikit-learn==1.4.2 xgboost joblib pandas numpy
python pipeline.py
```

This ingests the raw data, trains both Random Forest and XGBoost pipelines, saves train/test splits, evaluates both models, and exports the best one to `model/model_credit.joblib`.

Alternatively, run `train.ipynb` on a SageMaker notebook instance to execute the same steps interactively.

### Deploying to SageMaker

1. Open `deploy_endpoint.ipynb` on a SageMaker notebook instance (or an environment with `boto3` and `sagemaker` configured with appropriate AWS credentials/role).
2. Run the notebook to:
   * Compress `model/model_credit.joblib` into `model.tar.gz`
   * Upload the artifact to your S3 bucket
   * Deploy an `SKLearnModel` to a real-time endpoint using `inference.py` as the entry point
   * Test the endpoint with a sample payload
3. If a deployment attempt fails, use the cleanup cell in the notebook to delete the stale endpoint and endpoint configuration before retrying.

### Deploying the Streamlit App on EC2

1. Launch an EC2 instance (Amazon Linux) with an IAM role that has permission to invoke the SageMaker endpoint.
2. Paste the contents of `user-data.sh` into the instance's **User data** field at launch (update `GIT_REPO`, `ENDPOINT_NAME`, and `REGION` as needed).
3. On boot, the script will:
   * Install Python, pip, and git
   * Clone this repository
   * Create a virtual environment and install `requirements.txt`
   * Register and start a `systemd` service (`streamlit.service`) running `app_streamlit.py` on port 8501
4. Access the app at `http://<EC2-public-IP>:8501` (ensure the instance's security group allows inbound traffic on port 8501).

### Running the Streamlit App Locally (against a live endpoint)

```
pip install -r requirements.txt
export ENDPOINT_NAME=credit-score-endpoint
export AWS_REGION=us-east-1
streamlit run app_streamlit.py
```

Requires valid AWS credentials configured locally (e.g. via `aws configure`) with permission to invoke the SageMaker endpoint.
