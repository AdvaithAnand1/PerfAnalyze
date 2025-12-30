# record.py
"""
Record labeled telemetry data to CSV for training.

Usage examples (run while you perform the activity):
    python record.py idle 0.25
    python record.py gaming 0.25
    python record.py rendering 0.25
"""

import csv
import sys
import time
from datetime import datetime

from monitor import get_telemetry, FEATURE_NAMES


def record(label: str, out_csv: str = "training_data.csv", interval: float = 0.2):
    """
    Log telemetry + label at fixed interval until Ctrl+C.

    Each row: timestamp, label, <FEATURE_NAMES...>
    """
    sample = get_telemetry()
    n_feat = len(sample)

    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)

        # Write header if file is empty
        if f.tell() == 0:
            header = ["timestamp", "label"] + FEATURE_NAMES
            writer.writerow(header)

        print(f"Recording label='{label}' to {out_csv} every {interval}s (Ctrl+C to stop)")

        try:
            while True:
                x = get_telemetry()
                if x.shape[0] != n_feat:
                    raise RuntimeError("Feature length changed; check monitor.FEATURE_NAMES")
                row = [datetime.now().isoformat(), label, *x.tolist()]
                writer.writerow(row)
                f.flush()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped recording.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python record.py <label> [interval_seconds]")
        sys.exit(1)

    label_arg = sys.argv[1]
    interval_arg = float(sys.argv[2]) if len(sys.argv) >= 3 else 0.2
    record(label_arg, interval=interval_arg)
