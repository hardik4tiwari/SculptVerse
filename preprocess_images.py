import argparse
from openlrm.utils.preprocess import Preprocessor

def main():
    parser = argparse.ArgumentParser(description="Preprocess images.")
    parser.add_argument('input_path', type=str, help='Path to the input image.')
    parser.add_argument('output_path', type=str, help='Path to save the processed image.')
    parser.add_argument('--rmbg', action='store_true', help='Remove background.')
    parser.add_argument('--recenter', action='store_true', help='Recenter image.')
    parser.add_argument('--size', type=int, default=512, help='Size of the output image.')
    parser.add_argument('--border_ratio', type=float, default=0.2, help='Border ratio for recentering.')

    args = parser.parse_args()

    preprocessor = Preprocessor()
    preprocessor.preprocess(
        image_path=args.input_path,
        save_path=args.output_path,
        rmbg=args.rmbg,
        recenter=args.recenter,
        size=args.size,
        border_ratio=args.border_ratio
    )

if __name__ == "__main__":
    main()
