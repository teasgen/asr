import argparse
import os
from pathlib import Path

from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder import CTCTextEncoder

def calculate(pred_dir: str, gt_dir: str) -> dict[int, str]:
    path = Path(gt_dir)
    wers = []
    cers = []
    for file_abs_path in path.iterdir():
        original_name = file_abs_path.stem + ".txt"
        corresponding_predict_filename = os.path.join(pred_dir, original_name)
        pred = CTCTextEncoder.normalize_text(open(corresponding_predict_filename, "r").read())
        gt = CTCTextEncoder.normalize_text(open(file_abs_path, "r").read())
        cers.append(calc_cer(gt, pred))
        wers.append(calc_wer(gt, pred))
    return {
        "CER": sum(cers) / len(cers),
        "WER": sum(wers) / len(wers),
    }
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predicts-dir', type=str, help='directiry with prediction txt files')
    parser.add_argument('--gt-dir', type=str, help='directiry with gt txt files')
    args = parser.parse_args()

    results = calculate(args.predicts_dir, args.gt_dir)
    for x, y in results.items():
        print(x, y, sep=": ")