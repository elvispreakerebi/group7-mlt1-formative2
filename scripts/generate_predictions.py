from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_dataset, save_table
from src.inference import generate_final_predictions
from src.utils import ensure_output_dirs, load_config, resolve_path


def main() -> None:
    config = load_config()
    ensure_output_dirs(config)
    bundle = load_dataset(config)
    predictions, metadata = generate_final_predictions(bundle, config)
    prediction_path = f"{config['paths']['predictions_dir']}/devex_test_predictions.csv"
    metadata_path = f"{config['paths']['tables_dir']}/final_prediction_metadata.csv"
    save_table(predictions, prediction_path)
    save_table(metadata, metadata_path)
    print(f"Predictions written to {resolve_path(prediction_path)}")
    print(f"Final model metadata written to {resolve_path(metadata_path)}")


if __name__ == "__main__":
    main()
