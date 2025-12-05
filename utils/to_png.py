import os
import cv2
import numpy as np
import torch
import argparse

def to_png(input, output):

    # Check if input is file or directory
    if os.path.isdir(input):
        files = [os.path.join(input, file)
                 for file in os.listdir(input)
                 if not os.path.isdir(os.path.join(input, file))]
    elif os.path.isfile(input):
        files = [input]
    else:
        raise ValueError(f"Input path '{input}' is neither a file nor a directory")

    os.makedirs(output, exist_ok=True)

    for file in files:
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read image {file}, skipping.")
            continue

        img_tensor = torch.from_numpy(img.astype(np.float32))

        img_uint8 = img_tensor.numpy().astype(np.uint8)
        base_name = os.path.splitext(os.path.basename(file))[0]
        output_path = os.path.join(output, f"{base_name}.png")
        
        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, img_uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help="input image or folder")
    parser.add_argument('-o', '--output', type=str, help="output image")
    parser.add_argument('--verbose', action='store_true', help="verbose")
    args = parser.parse_args()
    
    if not args.input or not args.output:
        parser.error("No input and output has been provided")
    
    try:
        to_png(args.input, args.output)
    except Exception as e:
        print(f"{e}")

    if args.verbose:
        print(f"Image {args.input} has been saved as PNG image in {args.output}")