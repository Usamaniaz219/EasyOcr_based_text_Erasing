import cv2
import numpy as np
import math
import easyocr
import numpy as np


def load_image(image_path):
# Load the image
    return cv2.imread(image_path)

def calculate_num_rows_and_cols(image, tile_width, tile_height):
    # Calculate the number of rows and columns
    num_rows = math.ceil(image.shape[0] / tile_height)
    num_cols = math.ceil(image.shape[1] / tile_width)
    return num_rows, num_cols

def extract_tile(image, start_x, start_y, tile_width, tile_height):
    # Extract the tile from the image
    end_x = min(start_x + tile_width, image.shape[1])
    end_y = min(start_y + tile_height, image.shape[0])
    return image[start_y:end_y, start_x:end_x]





def replace_color_in_bbox(image, box):
    # Ensure there are exactly 4 points (for the minimum rotated rectangle)
    if len(box) != 4:
        raise ValueError("Exactly 4 points are required for the minimum rotated rectangle")
    
    # Create a binary mask for the rotated rectangle
    mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
    cv2.fillPoly(mask, [box], 255)
    
    # Get RGB values 2 pixels above the top-left corner
    try:
        min_x = np.min(box[:, 0])
        max_x = np.max(box[:, 0])
        min_y = np.min(box[:, 1])
        max_y = np.max(box[:, 1])
        
        if min_y - 2 < 0 or max_y + 2 >= image.shape[0] or min_x < 0 or max_x >= image.shape[1]:
            return

        rgb_above_top_left = image[min_y - 2, min_x]
    except IndexError:
        return
    
    # Get RGB values 2 pixels below the bottom-right corner
    try:
        rgb_below_bottom_right = image[max_y + 2, max_x]
    except IndexError:
        return
    
    # Check if the RGB values are approximately equal
    if np.allclose(rgb_above_top_left, rgb_below_bottom_right, atol=0.5):
        fill_color = rgb_above_top_left
        
        # Replace color in the masked area
        image[mask == 255] = fill_color
    else:
        pass

def detect_text_in_tile(image, tile_width, tile_height, reader):
    # Initialize a list to store the bounding box coordinates
    bounding_boxes = []
    output_image = np.copy(image)

    # Calculate number of rows and columns
    num_rows = image.shape[0] // tile_height
    num_cols = image.shape[1] // tile_width

    # Iterate over each row
    for r in range(num_rows):
        # Iterate over each column
        for c in range(num_cols):
            # Calculate the starting coordinates of the tile
            start_x = c * tile_width
            start_y = r * tile_height

            # Extract the tile from the image
            tile = image[start_y:start_y + tile_height, start_x:start_x + tile_width]

            # Perform text detection on the current tile using the detection model
            result = reader.readtext(tile, text_threshold=0.01)

            # Check if any bounding boxes were returned
            if len(result) > 0:
                # Extract the bounding box coordinates and text from the result
                for bbox, text, _ in result:
                    # Convert bbox to numpy array of type np.float32
                    bbox = np.array(bbox, dtype=np.float32)

                    # Get the four corner points of the minimum rotated rectangle
                    rect = cv2.minAreaRect(bbox)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)

                    # Adjust bounding box coordinates to fit the original image
                    box[:, 0] += start_x
                    box[:, 1] += start_y

                    # Append the rotated rectangle points to bounding_boxes
                    bounding_boxes.append(box.tolist())

                    # Create a binary mask for the rotated rectangle
                    mask = np.zeros_like(output_image[:, :, 0], dtype=np.uint8)
                    cv2.fillPoly(mask, [box], 255)
                    print("Type of box",type(box))

                    # Replace the detected text region with its RGB value or erase the text
                    # replace_color_in_bbox(output_image, box)

                    # Optionally, draw the rotated rectangle on the output image
                    cv2.polylines(output_image, [box], isClosed=True, color=(0, 0, 255), thickness=1)

                    # Print the detected text along with its coordinates
                    print(f'Text: "{text}" at coordinates: {box}')

    return bounding_boxes, output_image

def load_image(image_path):
    return cv2.imread(image_path)

def main(image_path, tile_width, tile_height):
    image = load_image(image_path)
    reader = easyocr.Reader(['en'], gpu=True)

    bounding_boxes, output_image = detect_text_in_tile(image, tile_width, tile_height, reader)

    return bounding_boxes, output_image

image_path = '/home/usama/Converted_jpg_from_tiff_july3_2024/ca_colma.jpg'
tile_width = 1024
tile_height = 1024

bounding_boxes, output_image = main(image_path, tile_width, tile_height)
print("Bounding box length:", len(bounding_boxes))
cv2.imwrite("/home/usama/EasyOCR_high_resolution_text_localization/text_erased_results/removed_text_7_2225.png", output_image)


