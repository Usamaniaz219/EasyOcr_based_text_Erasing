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

# def replace_color_in_bbox(image, bbox):
#     x1, y1 = int(bbox[0][0]), int(bbox[0][1])
#     print("X1 and y1 :",x1,y1)
#     x3, y3 = int(bbox[2][0]), int(bbox[2][1])
#     print("X3 and y3 :",x3,y3)
#     center_x = (x1 + x3) // 2
#     center_y = (y1 + y3) // 2
#     print("centerX and centerY :",center_x,center_y)
    
#     # Get RGB values 5 pixels above and below the center point
#     try:
#         rgb_above = image[center_y - 15, center_x]
#         print("RGB value Above:",rgb_above)
#         rgb_below = image[center_y + 15, center_x]
#         print("RGB value Below:",rgb_below)
#     except IndexError:
#         return
    
#     # Check if the RGB values are approximately equal
#     if np.allclose(rgb_above, rgb_below, atol=10):
#         fill_color = rgb_above
#         image[y1:y3, x1:x3] = fill_color
#     else:
#         # Fill upper region with rgb_above and lower region with rgb_below
#         image[y1:center_y, x1:x3] = rgb_above
#         image[center_y:y3, x1:x3] = rgb_below



# def replace_color_in_bbox(image, bbox):
#     x1, y1 = int(bbox[0][0]), int(bbox[0][1])
#     x3, y3 = int(bbox[2][0]), int(bbox[2][1])
    
#     # Get RGB values 5 pixels above the top-left corner
#     try:
#         rgb_above_top_left = image[y1 - 2, x1]
#         print("RGB value Above Top Left:", rgb_above_top_left)
#     except IndexError:
#         return
    
#     # Get RGB values 5 pixels below the bottom-right corner
#     try:
#         rgb_below_bottom_right = image[y3 + 2, x3]
#         print("RGB value Below Bottom Right:", rgb_below_bottom_right)
#     except IndexError:
#         return
    
#     # Check if the RGB values are approximately equal
#     if np.allclose(rgb_above_top_left, rgb_below_bottom_right, atol=0.2):
#         fill_color = rgb_above_top_left
#         image[y1:y3, x1:x3] = fill_color
#     else:
#         # Fill upper region with rgb_above_top_left and lower region with rgb_below_bottom_right
#         # center_y = (y1 + y3) // 2
#         # image[y1:center_y, x1:x3] = rgb_above_top_left
#         # image[center_y:y3, x1:x3] = rgb_below_bottom_right
#         pass


def replace_color_in_bbox(image, pts):
    # Ensure there are exactly 8 points
    if len(pts) != 8:
        raise ValueError("Exactly 8 points are required")

    x_coords = [pt[0][0] for pt in pts]
    y_coords = [pt[0][1] for pt in pts]

    # Calculate the bounding box as a rectangle
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)

    # Get RGB values 2 pixels above the top-left corner
    try:
        rgb_above_top_left = image[min_y - 2, min_x]
        print("RGB value Above Top Left:", rgb_above_top_left)
    except IndexError:
        return
    
    # Get RGB values 2 pixels below the bottom-right corner
    try:
        rgb_below_bottom_right = image[max_y + 2, max_x]
        print("RGB value Below Bottom Right:", rgb_below_bottom_right)
    except IndexError:
        return
    
    # Check if the RGB values are approximately equal
    if np.allclose(rgb_above_top_left, rgb_below_bottom_right, atol=0.015):
        fill_color = rgb_above_top_left
        image[min_y:max_y+1, min_x:max_x+1] = fill_color
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
                    try:
                        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = bbox
                    except ValueError:
                        continue

                    # Adjust bounding box coordinates to fit the original image
                    x1 += start_x
                    y1 += start_y
                    x2 += start_x
                    y2 += start_y
                    x3 += start_x
                    y3 += start_y
                    x4 += start_x
                    y4 += start_y

                    # Assuming 8 points are the diagonals and midpoints
                    mapped_bbox = [[x1, y1], [x2, y2], [x3, y3], [x4, y4],
                                   [(x1 + x2) // 2, (y1 + y2) // 2], [(x2 + x3) // 2, (y2 + y3) // 2],
                                   [(x3 + x4) // 2, (y3 + y4) // 2], [(x4 + x1) // 2, (y4 + y1) // 2]]
                    bounding_boxes.append(mapped_bbox)

                    # Draw tilted bounding box on the output image
                    pts = np.array(mapped_bbox, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    # cv2.polylines(output_image, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

                    # Replace the detected text with its RGB value or erase the text
                    replace_color_in_bbox(output_image, pts)

                    # Print the detected text along with its coordinates
                    print(f'Text: "{text}" at coordinates: {mapped_bbox}')

    return bounding_boxes, output_image

def load_image(image_path):
    return cv2.imread(image_path)

def main(image_path, tile_width, tile_height):
    image = load_image(image_path)
    reader = easyocr.Reader(['en'], gpu=True)

    bounding_boxes, output_image = detect_text_in_tile(image, tile_width, tile_height, reader)

    return bounding_boxes, output_image

image_path = '/home/usama/Converted_jpg_from_tiff_july3_2024/ca_dana_point.jpg'
tile_width = 1024
tile_height = 1024

bounding_boxes, output_image = main(image_path, tile_width, tile_height)
print("Bounding box length:", len(bounding_boxes))
cv2.imwrite("/home/usama/EasyOCR_high_resolution_text_localization/text_erased_results/removed_text_7_2.png", output_image)



# def replace_color_in_bbox(image, pts):
#     # Extract the bounding box coordinates from pts
#     if len(pts) <= 8:
#         raise ValueError("At least 8 points are required")
    
#     x_coords = [pt[0][0] for pt in pts]
#     y_coords = [pt[0][1] for pt in pts]
    
#     # Calculate the bounding box as a rectangle
#     min_x = min(x_coords)
#     max_x = max(x_coords)
#     min_y = min(y_coords)
#     max_y = max(y_coords)

#     # Get RGB values 2 pixels above the top-left corner
#     try:
#         rgb_above_top_left = image[min_y - 2, min_x]
#         print("RGB value Above Top Left:", rgb_above_top_left)
#     except IndexError:
#         return
    
#     # Get RGB values 2 pixels below the bottom-right corner
#     try:
#         rgb_below_bottom_right = image[max_y + 2, max_x]
#         print("RGB value Below Bottom Right:", rgb_below_bottom_right)
#     except IndexError:
#         return
    
#     # Check if the RGB values are approximately equal
#     if np.allclose(rgb_above_top_left, rgb_below_bottom_right, atol=0.015):
#         fill_color = rgb_above_top_left
#         image[min_y:max_y+1, min_x:max_x+1] = fill_color
#     else:
#         pass



# def detect_text_in_tile(image, tile_width, tile_height, reader):
#     # Initialize a list to store the bounding box coordinates
#     bounding_boxes = []
#     output_image = np.copy(image)

#     # Iterate over each row
#     num_rows, num_cols = calculate_num_rows_and_cols(image, tile_width, tile_height)
#     for r in range(num_rows):
#         # Iterate over each column
#         for c in range(num_cols):
#             # Calculate the starting coordinates of the tile
#             start_x = c * tile_width
#             start_y = r * tile_height

#             # Extract the tile from the image
#             tile = extract_tile(image, start_x, start_y, tile_width, tile_height)

#             # Perform text detection on the current tile using the detection model
#             result = reader.readtext(tile, text_threshold=0.01)

#             # Check if any bounding boxes were returned
#             if len(result) > 0:
#                 # Extract the bounding box coordinates and text from the result
#                 for bbox, text, _ in result:
#                     try:
#                         [[x1, y1], [x2, y2], [x3, y3], [x4, y4]] = bbox
#                     except ValueError:
#                         continue

#                     # Adjust bounding box coordinates to fit the original image
#                     x1 += start_x
#                     y1 += start_y
#                     x2 += start_x
#                     y2 += start_y
#                     x3 += start_x
#                     y3 += start_y
#                     x4 += start_x
#                     y4 += start_y

#                     mapped_bbox = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
#                     bounding_boxes.append(mapped_bbox)

#                     # Draw tilted bounding box on the output image
#                     pts = np.array(mapped_bbox, np.int32)
#                     pts = pts.reshape((-1, 1, 2))
#                     # cv2.polylines(output_image, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

#                     # Replace the detected text with its RGB value or erase the text
#                     replace_color_in_bbox(output_image, pts)

#                     # Print the detected text along with its coordinates
#                     print(f'Text: "{text}" at coordinates: {mapped_bbox}')

#     return bounding_boxes, output_image


# def main(image_path, tile_width, tile_height):
#     image = load_image(image_path)
#     reader = easyocr.Reader(['en'], gpu=True)

#     bounding_boxes, output_image = detect_text_in_tile(image, tile_width, tile_height, reader)

#     return bounding_boxes, output_image

# image_path = '/home/usama/Converted_jpg_from_tiff_july3_2024/ca_dana_point.jpg'
# tile_width = 1024
# tile_height = 1024

# bounding_boxes,output_image = main(image_path,tile_width,tile_height)
# print("bounding box length:",len(bounding_boxes))
# cv2.imwrite("/home/usama/EasyOCR_high_resolution_text_localization/text_erased_results/removed_text_7_2.png",output_image)
