import os
import glob
import json
import argparse
import subprocess
from src.utils.utils import calc_mean_score, save_json
from src.handlers.model_builder import Nima
from src.handlers.data_generator import TestDataGenerator


def convert_heic_to_jpeg_with_imagemagick(heic_path, output_dir):
    """Converts a HEIC image to JPEG using ImageMagick."""
    try:
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create the new file name with .jpg extension
        base_name = os.path.splitext(os.path.basename(heic_path))[0]
        jpg_path = os.path.join(output_dir, f"{base_name}.jpg")

        # Use ImageMagick's magick command instead of convert
        subprocess.run(['magick', 'convert', heic_path, jpg_path], check=True)

        print(f"Converted {heic_path} to {jpg_path}")

        # Optionally, delete the original HEIC file after conversion
        os.remove(heic_path)
        print(f"Deleted original HEIC file: {heic_path}")

        return jpg_path
    except subprocess.CalledProcessError as e:
        print(f"Failed to convert {heic_path}: {e}")
        return None
def convert_heic_images_in_directory(img_dir):
    """Converts all HEIC images in the directory to JPEG using ImageMagick."""
    heic_paths = glob.glob(os.path.join(img_dir, '*.heic'))
    converted_files = []
    for heic_path in heic_paths:
        converted_img_path = convert_heic_to_jpeg_with_imagemagick(heic_path, img_dir)
        if converted_img_path:
            converted_files.append(converted_img_path)
    return converted_files


def image_file_to_json(img_path):
    img_dir = os.path.dirname(img_path)
    img_id = os.path.basename(img_path).split('.')[0]
    return img_dir, [{'image_id': img_id}]


def image_dir_to_json(img_dir, img_types=['jpg', 'jpeg']):
    img_paths = []
    # Loop through all image types (jpg, jpeg, etc.)
    for img_type in img_types:
        img_paths.extend(glob.glob(os.path.join(img_dir, f'*.{img_type}')))

    samples = []
    for img_path in img_paths:
        img_id = os.path.basename(img_path).split('.')[0]
        samples.append({'image_id': img_id})
    return samples


def predict(model, data_generator):
    return model.predict(data_generator, verbose=1)

def main(base_model_name, weights_file, image_source, predictions_file, img_format='jpg'):
    # Convert HEIC images to JPEG if the source is a directory
    if os.path.isdir(image_source):
        print(f"Checking {image_source} for HEIC images to convert...")
        converted_files = convert_heic_images_in_directory(image_source)
        if converted_files:
            print(f"Converted {len(converted_files)} HEIC files to JPEG.")

    # Load samples
    if os.path.isfile(image_source):
        image_dir, samples = image_file_to_json(image_source)
    else:
        image_dir = image_source
        samples = image_dir_to_json(image_dir)

    # Build model and load weights
    nima = Nima(base_model_name, weights=None)
    nima.build()
    nima.nima_model.load_weights(weights_file)

    # Initialize data generator
    data_generator = TestDataGenerator(samples, image_dir, 64, 10, nima.preprocessing_function(), img_format=img_format)

    # Get predictions
    predictions = predict(nima.nima_model, data_generator)

    # Calculate mean scores and add to samples
    for i, sample in enumerate(samples):
        sample['mean_score_prediction'] = calc_mean_score(predictions[i])

    print(json.dumps(samples, indent=2))

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
