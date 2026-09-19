from pathlib import Path
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

class CreditScorePreprocessor:

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def clean_numeric_string(self, value):
        if pd.isna(value):
            return np.nan

        value = str(value).strip()

        if value == "" or value.replace("_", "") == "":
            return np.nan

        value = value.replace("_", "")

        try:
            return float(value)
        except:
            return np.nan

    def convert_credit_history_age(self, x):
        if pd.isna(x):
            return np.nan

        years = re.search(r"(\d+)\s*Years?", str(x))
        months = re.search(r"(\d+)\s*Months?", str(x))

        y = int(years.group(1)) if years else 0
        m = int(months.group(1)) if months else 0

        return y * 12 + m

    def clean_data(self, data_path):
        df = pd.read_csv(Path(data_path))

        df["Occupation"] = df["Occupation"].replace("_______", np.nan)
        df["Credit_Mix"] = df["Credit_Mix"].replace("_", np.nan)
        df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].replace("NM", np.nan)
        df["Payment_Behaviour"] = df["Payment_Behaviour"].replace("!@9#%8", np.nan)
        df["Amount_invested_monthly"] = df["Amount_invested_monthly"].replace("__10000__", np.nan)
        df["Monthly_Balance"] = df["Monthly_Balance"].replace("__-333333333333333333333333333__", np.nan)

        numeric_as_object_cols = ["Age",
                                  "Annual_Income",
                                  "Num_of_Loan",
                                  "Num_of_Delayed_Payment",
                                  "Changed_Credit_Limit",
                                  "Outstanding_Debt",
                                  "Amount_invested_monthly",
                                  "Monthly_Balance"]
        for col in numeric_as_object_cols:
            df[col] = df[col].apply(self.clean_numeric_string)
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["Credit_History_Age_Months"] = (df["Credit_History_Age"].apply(self.convert_credit_history_age))
        df.drop(columns=["Credit_History_Age"], inplace=True)

        df["Computed_Num_of_Loan"] = df["Type_of_Loan"].apply(
            lambda x: len([i for i in x.split(", ") if i.strip()])
            if pd.notnull(x) else np.nan
        )
        invalid_mask = ((df["Num_of_Loan"] < 0) | (df["Num_of_Loan"] > 50))
        df.loc[invalid_mask, "Num_of_Loan"] = np.nan
        df["Num_of_Loan"] = df["Num_of_Loan"].fillna(df["Computed_Num_of_Loan"])
        df = df.drop(columns=['Type_of_Loan', 'Computed_Num_of_Loan'])

        df.loc[(df["Age"] < 18) | (df["Age"] > 100), "Age"] = np.nan
        df.loc[(df["Num_Bank_Accounts"] < 0) | (df["Num_Bank_Accounts"] > 20), "Num_Bank_Accounts"] = np.nan
        df.loc[df["Num_Credit_Card"] > 20, "Num_Credit_Card"] = np.nan
        df.loc[df["Interest_Rate"] > 60, "Interest_Rate"] = np.nan
        df.loc[df["Delay_from_due_date"] < 0, "Delay_from_due_date"] = np.nan
        df.loc[(df["Num_of_Delayed_Payment"] < 0) | (df["Num_of_Delayed_Payment"] > 60), "Num_of_Delayed_Payment"] = np.nan
        df.loc[df["Changed_Credit_Limit"] < 0, "Changed_Credit_Limit"] = np.nan
        df.loc[df["Num_Credit_Inquiries"] > 50, "Num_Credit_Inquiries"] = np.nan

        df.drop(columns=["Unnamed: 0", "ID", "Customer_ID", "Name", "SSN"], inplace=True, errors="ignore")

        df["Annual_Income_log"] = np.log1p(df["Annual_Income"])
        df.drop(columns=["Annual_Income"], inplace=True)
        df.drop(columns=["Month", "Occupation"], inplace=True)

        label_mapping = {"Poor": 0, "Standard": 1, "Good": 2}
        df["Credit_Score"] = df["Credit_Score"].map(label_mapping)

        return df

    def split_data(self, df):
        x = df.drop(columns=["Credit_Score"])
        y = df["Credit_Score"]

        return train_test_split(x, y, test_size=self.test_size, random_state=self.random_state, stratify=y)

    def get_transformer(self, x_train):
        num_cols = x_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
        num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        credit_mix_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                        ("encoder", OrdinalEncoder(categories=[["Bad", "Standard", "Good"]]))])
        payment_min_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), 
                                        ("encoder", OrdinalEncoder(categories=[["No", "Yes"]]))])
        payment_behaviour_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                               ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))])

        transformer = ColumnTransformer([("num", num_pipeline, num_cols),
                                         ("credit_mix", credit_mix_pipeline, ["Credit_Mix"]),
                                         ("payment_min", payment_min_pipeline, ["Payment_of_Min_Amount"]),
                                         ("payment_behaviour", payment_behaviour_pipeline, ["Payment_Behaviour"])])

        return transformer
    
    def save_split_data(self, x_train, x_test, y_train, y_test, train_dir="train", test_dir="test"):
        train_dir = Path(train_dir)
        test_dir = Path(test_dir)

        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        train_df = pd.concat([x_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
        test_df = pd.concat([x_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

        train_df.to_csv(train_dir / "train.csv", index=False)
        test_df.to_csv(test_dir / "test.csv", index=False)

        print("Train data saved to train/train.csv")
        print("Test data saved to test/test.csv")