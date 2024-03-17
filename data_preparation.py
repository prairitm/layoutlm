# Functions related to data loading and preprocessing.
# Includes the JSON import section, hocr_to_dataframe function, and any other preprocessing steps.

# data_preparation.py

import os
import json
import pandas as pd
import glob
from config import INPUT_IMAGES_DIR, LABEL_STUDIO_JSON_PATH, LAYOUTLMV3_HOCR_OUTPUT_DIR
from utils import hocr_to_dataframe, calculate_iou

def load_label_studio_json(json_path):
    """
    Load and return the content of the Label Studio JSON output file.
    """
    with open(json_path) as f:
        label_studio_data = json.load(f)
    return label_studio_data

def extract_document_data(label_studio_data):
    """
    Process Label Studio JSON data to extract relevant information for each document.
    """
    document_data = {'file_name': [], 'labelled_bbox': []}

    for entry in label_studio_data:
        file_name = os.path.basename(entry['data']['image'])
        label_list = []

        for annotation in entry['annotations'][0]['result']:
            label_value = annotation['value']
            x, y, w, h = label_value['x'], label_value['y'], label_value['width'], label_value['height']
            original_w, original_h = annotation['original_width'], annotation['original_height']

            x1 = int((x * original_w) / 100)
            y1 = int((y * original_h) / 100)
            x2 = x1 + int(original_w * w / 100)
            y2 = y1 + int(original_h * h / 100)

            label = label_value['rectanglelabels'][0]
            label_list.append((label, (x1, y1, x2, y2), original_h, original_w))

        document_data['file_name'].append(file_name)
        document_data['labelled_bbox'].append(label_list)

    return pd.DataFrame(document_data)

def generate_custom_dataset(input_dir, label_studio_json_path):
    """
    Generate a custom dataset by processing the labeled data and input images.
    """
    label_studio_data = load_label_studio_json(label_studio_json_path)
    custom_dataset = extract_document_data(label_studio_data)

    final_list = []

    for _, row in custom_dataset.iterrows():
        file_name = row['file_name']
        label_list = row['labelled_bbox']
        image_path = os.path.join(input_dir, file_name)

        if os.path.exists(image_path):
            # Placeholder for any processing you need to do per image, such as feature extraction or OCR
            # For example: custom_label_text = process_image(image_path, label_list, label2id)
            # final_list.append(custom_label_text)
            pass

    return final_list

def main():
    """
    Main function to execute the data preparation steps.
    """
    custom_dataset = generate_custom_dataset(INPUT_IMAGES_DIR, LABEL_STUDIO_JSON_PATH)
    # Here, you would continue to process `custom_dataset` as needed for your model training

if __name__ == "__main__":
    main()
