# Auto-generated file
import csv
import os
import pandas as pd


class CSVLogger:

    def __init__(self, path):

        self.path = path
        
        self.writer = None

        self.last_step = -1
        self.cum_pred_first = 0.0
        self.cum_pred_second = 0.0
        self.cum_pred_full = 0.0
        self.cum_actual = 0.0
        
        dir_name = os.path.dirname(path)

        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        self.file = open(path, "a", newline="")
        if os.path.exists(path) and os.path.getsize(path) > 0:

            df = pd.read_csv(path)

            self.last_step = int(df["step"].iloc[-1])

            self.cum_pred_first = df["cum_pred_first"].iloc[-1]
            self.cum_pred_second = df["cum_pred_second"].iloc[-1]
            self.cum_pred_full = df["cum_pred_full"].iloc[-1]
            self.cum_actual = df["cum_actual"].iloc[-1]

    def log(self, row):

        if self.writer is None:

            file_empty = os.stat(self.path).st_size == 0

            self.writer = csv.DictWriter(
                self.file,
                fieldnames=list(row.keys())
            )

            if file_empty:
                self.writer.writeheader()

        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()