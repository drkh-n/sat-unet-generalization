from tqdm import tqdm
import argparse
import logging
from utils.segmentator import Segmentator


def main() -> None:
    parser = argparse.ArgumentParser(description="Segmentation Inference Script")
    parser.add_argument("--data_path", "-d", type=str, default="./data/",
                        help="Path to input image folder.")
    parser.add_argument("--hparams", "-p", type=str, default="./configs/satunet.yml",
                        help="Path to YAML hyperparameters config.")
    parser.add_argument("--ckpt", "-c", type=str, default="./checkpoints/satunet/version_0/epoch=00.ckpt",
                        help="Path to checkpoint file.")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                        help="Optional threshold value for predictions.")
    parser.add_argument("--save_dir", "-s", type=str, default="./samples/",
                        help="Directory to save results.")

    args = parser.parse_args()

    predictor = Segmentator(args.hparams, args.ckpt)

    ids = predictor.get_all_ids(args.data_path)
    logging.info(f"Found {len(ids)} images to process.")

    for image_id in tqdm(ids):
        batch = predictor.get_single_batch(args.data_path, image_id)
        outputs = predictor.predict(batch, threshold=args.threshold)
        save_path = f"{args.save_dir}/{image_id}.png"
        predictor.save_prediction(outputs, save_path)


if __name__ == "__main__":
    main()
