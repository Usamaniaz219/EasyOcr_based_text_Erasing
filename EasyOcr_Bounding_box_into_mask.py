import cv2
import numpy as np
import math
import easyocr
import os 

def load_image(image_path):
    return cv2.imread(image_path)

def calculate_num_rows_and_cols(image, tile_width, tile_height):
    num_rows = math.ceil(image.shape[0] / tile_height)
    num_cols = math.ceil(image.shape[1] / tile_width)
    return num_rows, num_cols

def extract_tile(image, start_x, start_y, tile_width, tile_height):
    end_x = min(start_x + tile_width, image.shape[1])
    end_y = min(start_y + tile_height, image.shape[0])
    return image[start_y:end_y, start_x:end_x]

def create_mask_for_bboxes(image_shape, bounding_boxes):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for box in bounding_boxes:
        box = np.array(box, dtype=np.int32)
        cv2.fillPoly(mask, [box], 255)
    return mask

def detect_text_in_tile(image, tile_width, tile_height, reader):
    bounding_boxes = []
    num_rows, num_cols = calculate_num_rows_and_cols(image, tile_width, tile_height)

    for r in range(num_rows):
        for c in range(num_cols):
            start_x = c * tile_width
            start_y = r * tile_height
            tile = extract_tile(image, start_x, start_y, tile_width, tile_height)

            result = reader.readtext(tile, text_threshold=0.01)

            if len(result) > 0:
                for bbox, text, _ in result:
                    bbox = np.array(bbox, dtype=np.float32)
                    rect = cv2.minAreaRect(bbox)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    box[:, 0] += start_x
                    box[:, 1] += start_y
                    bounding_boxes.append(box.tolist())

    return bounding_boxes

def main(image_path, tile_width, tile_height):
    image = load_image(image_path)
    reader = easyocr.Reader(['en'], gpu=True)

    bounding_boxes = detect_text_in_tile(image, tile_width, tile_height, reader)
    mask = create_mask_for_bboxes(image.shape, bounding_boxes)

    return bounding_boxes, mask

# image_path = '/home/usama/Converted_jpg_from_tiff_july3_2024/ca_colma.jpg'
# tile_width = 1024
# tile_height = 1024

# bounding_boxes, mask = main(image_path, tile_width, tile_height)
# print("Bounding box length:", len(bounding_boxes))
# cv2.imwrite("/home/usama/EasyOCR_high_resolution_text_localization/text_erased_results/mask.png", mask)


# image_path = '/home/usama/Converted_jpg_from_tiff_july3_2024/ca_colma.jpg'
directory = '/home/usama/Converted_jpg_from_tiff_july3_2024/'
output_directory = '/home/usama/EasyOCR_high_resolution_text_localization/text_masks_results_july_13_2024/'
os.makedirs(output_directory, exist_ok=True)
image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jpg') or f.endswith('.png')]

tile_width = 1024
tile_height = 1024
for image_path in image_paths:

    bounding_boxes, output_image = main(image_path, tile_width, tile_height)
    output_filename = os.path.basename(image_path).replace('.jpg', 'text_mask.png').replace('.png', 'text_mask.png')
    output_path = os.path.join(output_directory, output_filename)
    cv2.imwrite(output_path, output_image)
