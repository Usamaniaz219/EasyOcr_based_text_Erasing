import cv2
import numpy as np
import math
import easyocr
import numpy as np

from collections import Counter
import os


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


# def replace_color_in_bbox(image, box, combined_mask):
#     # Ensure there are exactly 4 points (for the minimum rotated rectangle)
#     if len(box) != 4:
#         raise ValueError("Exactly 4 points are required for the minimum rotated rectangle")
    
#     # Create a binary mask for the rotated rectangle
#     mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
#     cv2.fillPoly(mask, [box], 255)
    
#     # Update the combined mask
#     combined_mask = np.bitwise_or(combined_mask, mask)
    
#     # Get RGB values 2 pixels above the top-left corner
#     try:
#         min_x = np.min(box[:, 0])
#         max_x = np.max(box[:, 0])
#         min_y = np.min(box[:, 1])
#         max_y = np.max(box[:, 1])
        
#         if min_y - 2 < 0 or max_y + 2 >= image.shape[0] or min_x < 0 or max_x >= image.shape[1]:
#             return combined_mask, None

#         rgb_above_top_left = image[min_y - 2, min_x]
#     except IndexError:
#         return combined_mask, None
    
#     # Get RGB values 2 pixels below the bottom-right corner
#     try:
#         rgb_below_bottom_right = image[max_y + 2, max_x]
#     except IndexError:
#         return combined_mask, None
    
#     # Check if the RGB values are approximately equal
#     if np.allclose(rgb_above_top_left, rgb_below_bottom_right, atol=50):
#         fill_color = rgb_above_top_left
#         print("Type of fill color:",type(fill_color))
#     else:
#         fill_color = np.array([0,255,0])
    
#     return combined_mask, fill_color



# 



# def get_most_frequent_color(neighbors):
#     """ Returns the most frequent RGB color from the list of neighbors. """
#     if not neighbors:
#         return None
#     return Counter(tuple(color) for color in neighbors).most_common(1)[0][0]

# def replace_color_in_bbox(image, box, combined_mask):
#     if len(box) != 4:
#         raise ValueError("Exactly 4 points are required for the minimum rotated rectangle")

#     # Create a binary mask for the rotated rectangle
#     mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
#     cv2.fillPoly(mask, [box], 255)
#     combined_mask = np.bitwise_or(combined_mask, mask)

#     # Get the bounding box coordinates
#     min_x = int(np.min(box[:, 0]))
#     max_x = int(np.max(box[:, 0]))
#     min_y = int(np.min(box[:, 1]))
#     max_y = int(np.max(box[:, 1]))

#     # Collect RGB values from the 4 neighbors for every point in the bounding box
#     neighbors = []
    
#     for y in range(min_y, max_y + 1):
#         for x in range(min_x, max_x + 1):
#             # Collect neighbors (left, right, above, below)
#             neighbor_coords = [
#                 (y - 1, x),   # above
#                 (y + 1, x),   # below
#                 (y, x - 1),   # left
#                 (y, x + 1)    # right
#             ]
            
#             for ny, nx in neighbor_coords:
#                 if 0 <= ny < image.shape[0] and 0 <= nx < image.shape[1]:
#                     neighbors.append(image[ny, nx])

#     # Find the most frequent color among the neighbors
#     fill_color = get_most_frequent_color(neighbors)

#     if fill_color is None:
#         fill_color = np.array([0, 255, 0])  # Default color if no neighbors are found

#     return combined_mask, np.array(fill_color)



def get_most_frequent_color(neighbors):
    """ Returns the most frequent RGB color from the list of neighbors. """
    if not neighbors:
        return None
    return Counter(tuple(color) for color in neighbors).most_common(1)[0][0]

def replace_color_in_bbox(image, box, combined_mask):
    print("value of box is :",box)
    if len(box) != 4:
        raise ValueError("Exactly 4 points are required for the minimum rotated rectangle")

    # Create a binary mask for the rotated rectangle
    mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
    cv2.fillPoly(mask, [box], 255)
    combined_mask = np.bitwise_or(combined_mask, mask)


#     neighbors = []

#     def get_pixel_and_neighbors(image, y, x):
#         height, width, _ = image.shape
#         if 0 <= y < height and 0 <= x < width:
#             neighbors.append(image[y, x])
#             neighbor_coords = [
#                 (y - 1, x),   # above
#                 (y + 1, x),   # below
#                 (y, x - 1),   # left
#                 (y, x + 1)    # right
#             ]

#             for ny, nx in neighbor_coords:
#                 if 0 <= ny < height and 0 <= nx < width:
#                     neighbors.append(image[ny, nx])

# # Loop through each coordinate in the box
#     for coord in box:
#         x, y = coord
#         y_above = y - 2  # 2 pixels above
#         get_pixel_and_neighbors(image, y_above, x)



#     fill_color = get_most_frequent_color(neighbors)

#     if fill_color is None:
#         fill_color = np.array([0, 255, 0])  # Default color if no neighbors are found
    neighbors = []

    # Function to get pixel value and its neighbors
    def get_pixel_and_neighbors(image, y, x):
        height, width, _ = image.shape
        pixels = []
        if 0 <= y < height and 0 <= x < width:
            pixels.append(tuple(image[y, x]))
            neighbor_coords = [
                (y - 1, x),   # above
                (y + 1, x),   # below
                (y, x - 1),   # left
                (y, x + 1)    # right
            ]

            for ny, nx in neighbor_coords:
                if 0 <= ny < height and 0 <= nx < width:
                    pixels.append(tuple(image[ny, nx]))
        return pixels

    # Loop through each coordinate in the box
    # for coord in box:
    #     x, y = coord
    #     y_above = y - 2  # 2 pixels above
    #     neighbors.extend(get_pixel_and_neighbors(image, y_above, x))
    top_left_x, top_left_y = box[0]
    y_above = top_left_y + 2  # 2 pixels below
    neighbors.extend(get_pixel_and_neighbors(image, y_above,top_left_x ))

    top_left_freq = Counter(neighbors[:len(box) * 5]) 
    top_left_most_common = top_left_freq.most_common(1)[0][0] 
    # Get the bottom-right corner and 2 pixels below it
    bottom_right_x, bottom_right_y = box[-1]
    y_below = bottom_right_y + 2  # 2 pixels below
    neighbors.extend(get_pixel_and_neighbors(image, y_below, bottom_right_x))

    # Calculate the most frequent values
    # top_left_freq = Counter(neighbors[:len(box) * 5])  # 5 pixels (center + 4 neighbors) for each top-left coordinate
    bottom_right_freq = Counter(neighbors[len(box) * 5:])  # 5 pixels for bottom-right corner and its neighbors
    print("Bottom right frequency:",len(bottom_right_freq))

    # top_left_most_common = top_left_freq.most_common(1)[0][0]
    bottom_right_most_common = bottom_right_freq.most_common(1)[0][0]

    # Calculate the difference and apply threshold
    threshold = 2  # Example threshold
    difference = np.abs(np.array(top_left_most_common) - np.array(bottom_right_most_common))

    if np.all(difference < threshold):
        fill_color = np.array(top_left_most_common)
    else:
        fill_color = [0,255,0]

    return combined_mask, np.array(fill_color)




def detect_text_in_tile(image, tile_width, tile_height, reader):
    # Initialize a list to store the bounding box coordinates
    bounding_boxes = []
    output_image = np.copy(image)
    
    # Create a combined mask to keep track of all masked areas
    combined_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)

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
            result = reader.readtext(tile, text_threshold=0.6)

            # Check if any bounding boxes were returned
            if len(result) > 0:
                # Extract the bounding box coordinates and text from the result
                for bbox, text, _ in result:
                    # Convert bbox to numpy array of type np.float32
                    bbox = np.array(bbox, dtype=np.float32)
                    # print("bbox value is:",bbox)
                    # Get the four corner points of the minimum rotated rectangle
                    rect = cv2.minAreaRect(bbox)
                    box = cv2.boxPoints(rect)
                    # print("Value of box:",box)
                    box = np.int0(box)

                    # Adjust bounding box coordinates to fit the original image
                    box[:, 0] += start_x
                    box[:, 1] += start_y

                    # Append the rotated rectangle points to bounding_boxes
                    bounding_boxes.append(box.tolist())
                   

                    # Replace the detected text region with its RGB value or erase the text
                    combined_mask, fill_color = replace_color_in_bbox(output_image, box, combined_mask)
                    print("Type of  fill Color is :",type(fill_color))

                    # Optionally, draw and fill the rotated rectangle on the output image
                    if fill_color is not None:
                        cv2.fillPoly(output_image, [box], color=fill_color.tolist())
                        # cv2.fillPoly(output_image, [box], color=fill_color)
                    else:
                        cv2.polylines(output_image, [box], isClosed=True, color=(0, 0, 255), thickness=1)

                    # Print the detected text along with its coordinates
                    # print(f'Text: "{text}" at coordinates: {box}')

    # Apply the combined mask to the output image
    output_image[combined_mask == 255] = output_image[combined_mask == 255]
    
    return bounding_boxes, output_image


def load_image(image_path):
    return cv2.imread(image_path)

def main(image_path, tile_width, tile_height):
    image = load_image(image_path)
    reader = easyocr.Reader(['en'], gpu=True)

    bounding_boxes, output_image = detect_text_in_tile(image, tile_width, tile_height, reader)

    return bounding_boxes, output_image

# image_path = '/home/usama/Converted_jpg_from_tiff_july3_2024/ca_colma.jpg'
directory = '/home/usama/Converted_1_jpg_from_tiff_july3_2024/'
output_directory = '/home/usama/EasyOCR_high_resolution_text_localization/text_results_erased_july_12_2024/'
os.makedirs(output_directory, exist_ok=True)
image_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.jpg') or f.endswith('.png')]

tile_width = 1024
tile_height = 1024
for image_path in image_paths:

    bounding_boxes, output_image = main(image_path, tile_width, tile_height)
    output_filename = os.path.basename(image_path).replace('.jpg', '_removed_text.png').replace('.png', '_removed_text.png')
    output_path = os.path.join(output_directory, output_filename)
    cv2.imwrite(output_path, output_image)


    # cv2.imwrite("/home/usama/EasyOCR_high_resolution_text_localization/text_erased_results/ca_dana_point_removed_text_7_222.png", output_image)

