from pathlib import Path
import pandas as pd

class DataIngestion:

    def __init__(self, input_path, output_dir):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_file = self.output_dir / "credit_score.csv"

    def run(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(self.input_path)

        assert not df.empty, "Dataset is empty"

        df.to_csv(self.output_file, index=False)
        print(f"Data saved to {self.output_file}")

        return self.output_file