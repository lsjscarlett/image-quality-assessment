import os
import glob
import json
import argparse
import subprocess
from src.utils.utils import calc_mean_score, save_json
from src.handlers.model_builder import Nima
from src.handlers.data_generator import TestDataGenerator
from PIL import Image


def verify_image(image_path):
    """Verify if an image is valid."""
    try:
        img = Image.open(image_path)
        img.verify()  # Check if the image is corrupted
        print(f"Verified image: {image_path}")
    except (IOError, SyntaxError) as e:
        print(f"Corrupted image: {image_path} - {e}")
        return False
    return True


def correct_file_extension(image_path):
    """Detect if the file extension is incorrect based on the actual file type and rename."""
    try:
        # Use the `file` command to detect the real format
        result = subprocess.run(['file', '--brief', '--mime-type', image_path], stdout=subprocess.PIPE)
        mime_type = result.stdout.decode().strip()

        # Rename based on the mime type
        if mime_type == 'image/heic' or mime_type == 'image/heif':
            base_name, _ = os.path.splitext(image_path)
            correct_path = f"{base_name}.heic"
            os.rename(image_path, correct_path)
            print(f"Renamed {image_path} to {correct_path} (HEIC format)")
            return correct_path
        elif mime_type == 'image/jpeg':
            return image_path  # Already correct
        else:
            print(f"Unknown or unsupported format for {image_path}: {mime_type}")
            return image_path
    except Exception as e:
        print(f"Error detecting file type for {image_path}: {e}")
        return None


def convert_to_jpeg(img_path):
    """Convert any image to JPEG if it's not already in the correct format."""
    img_path = correct_file_extension(img_path)
    if not img_path:
        return None

    try:
        if img_path.endswith('.heic'):
            # Convert HEIC to JPEG using ImageMagick
            base_name = os.path.splitext(img_path)[0]
            jpg_path = f"{base_name}.jpg"
            subprocess.run(['magick', 'convert', img_path, jpg_path], check=True)
            print(f"Converted {img_path} to JPEG using ImageMagick: {jpg_path}")
            os.remove(img_path)  # Delete the original HEIC file
            return jpg_path
        else:
            # Use PIL for other formats
            with Image.open(img_path) as img:
                if img.format != 'JPEG':
                    base_name = os.path.splitext(img_path)[0]
                    jpg_path = f"{base_name}.jpg"
                    img.convert("RGB").save(jpg_path, "JPEG")
                    print(f"Converted {img_path} to JPEG: {jpg_path}")
                    os.remove(img_path)  # Delete original non-JPEG file
                    return jpg_path
                else:
                    print(f"Image {img_path} is already in JPEG format.")
                    return img_path
    except Exception as e:
        print(f"Error converting {img_path}: {e}")
        return None


def check_and_convert_images_in_directory(img_dir):
    """Check and convert any images with incorrect formats."""
    img_paths = glob.glob(os.path.join(img_dir, '*.*'))  # Get all images, regardless of extension
    valid_files = []

    for img_path in img_paths:
        converted_img_path = convert_to_jpeg(img_path)
        if converted_img_path and verify_image(converted_img_path):
            valid_files.append(converted_img_path)

    return valid_files


def image_file_to_json(img_path):
    """Generate JSON metadata from a single image file."""
    img_dir = os.path.dirname(img_path)
    img_id = os.path.basename(img_path).split('.')[0]
    return img_dir, [{'image_id': img_id}]


def image_dir_to_json(img_dir):
    """Generate JSON metadata from all images in a directory."""
    img_paths = check_and_convert_images_in_directory(img_dir)
    print(f"Images found: {len(img_paths)}")

    samples = []
    for img_path in img_paths:
        img_id = os.path.basename(img_path).split('.')[0]
        samples.append({'image_id': img_id})
    return samples


def predict(model, data_generator):
    """Run the prediction using the model and data generator."""
    return model.predict(data_generator, verbose=1)


def main(base_model_name, weights_file, image_source, predictions_file, img_format='jpg'):
    """Main pipeline for handling image prediction."""
    # Check and convert images if the source is a directory
    if os.path.isdir(image_source):
        print(f"Checking {image_source} for improperly formatted images...")
        samples = image_dir_to_json(image_source)
    elif os.path.isfile(image_source):
        image_dir, samples = image_file_to_json(image_source)
    else:
        print(f"Error: {image_source} is neither a valid file nor a directory.")
        return

    # Build model and load weights
    nima = Nima(base_model_name, weights=None)
    nima.build()
    nima.nima_model.load_weights(weights_file)

    # Initialize data generator
    data_generator = TestDataGenerator(samples, image_source, 64, 10, nima.preprocessing_function(),
                                       img_format=img_format)

    # Get predictions
    predictions = predict(nima.nima_model, data_generator)

    # Calculate mean scores and add to samples
    for i, sample in enumerate(samples):
        sample['mean_score_prediction'] = calc_mean_score(predictions[i])

    # Output the results in JSON format
    print(json.dumps(samples, indent=2))

    # Save predictions to a file if provided
    if predictions_file is not None:
        save_json(samples, predictions_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base-model-name', help='CNN base model name', required=True)
    parser.add_argument('-w', '--weights-file', help='path of weights file', required=True)
    parser.add_argument('-is', '--image-source', help='image directory or file', required=True)
    parser.add_argument('-pf', '--predictions-file', help='file with predictions', required=False, default=None)
    args = parser.parse_args()
    main(**args.__dict__)
