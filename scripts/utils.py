import csv
import os

def log_results(file_path, row):
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "round","client_id","loss","accuracy",
                "f1","precision","recall","lambda","gate_rate"
            ])

        writer.writerow(row)