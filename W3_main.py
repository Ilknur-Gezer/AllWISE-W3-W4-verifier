# main.py

import sys
from W3_model_utils import train_models, predict_data

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|predict]")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "train":
        train_models()
    elif command == "predict":
        predict_data()
    else:
        print(f"Unknown command: {command}")
        print("Usage: python main.py [train|predict]")
