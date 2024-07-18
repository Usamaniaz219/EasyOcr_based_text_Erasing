
# import numpy as np
# import easyocr
# import cv2
# import math

# def draw_bounding_boxes(mask, bounding_boxes):
#     # Create a zero mask image to draw bounding boxes on
#     # Convert mask image into binary such that foreground pixels are white and background pixels are black
#     _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
#     bbox_mask = np.zeros_like(mask, dtype=np.uint8)

#     # Iterate through each bounding box
#     for bbox in bounding_boxes:
#         # Convert bounding box coordinates to integers
#         bbox = np.array(bbox, dtype=np.int32)

#         # Draw the bounding box on the zero mask image
#         cv2.fillPoly(bbox_mask, [bbox], color=(255))

#     # This retains only the bounding boxes that overlap with the foreground
#     result_mask = cv2.bitwise_and(mask, bbox_mask)
#     # _, result_mask = cv2.threshold(result_mask, 127, 255, cv2.THRESH_BINARY)
#     result_mask = cv2.cvtColor(result_mask, cv2.COLOR_BGR2GRAY)


#     # Finding contours directly from the binary mask
#     contours, _ = cv2.findContours(result_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Create a blank mask to draw filled contours
#     filled_mask = np.zeros_like(mask, dtype=np.uint8)

#     # Draw filled contours on the blank mask
#     cv2.drawContours(filled_mask, contours, -1, (255), thickness=cv2.FILLED)
#     for contour in contours:       
#             cv2.fillPoly(mask, [contour], (255))
#     # convert mask image into gray image
#     # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    

#     return mask

import numpy as np
import easyocr
import cv2
import math



def draw_bounding_boxes(mask, bounding_boxes, image):
    # Create a zero mask image to draw bounding boxes on
    # Convert mask image into binary such that foreground pixels are white and background pixels are black
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask,(3,3),0)
    # bbox_mask = np.zeros_like(mask, dtype=np.uint8)
    bbox_mask = mask.copy()

    # Iterate through each bounding box
    for bbox in bounding_boxes:
        # Convert bounding box coordinates to integers
        bbox = np.array(bbox, dtype=np.int32)
        # bbox = list(bbox,dtype = np.int32)

        # Draw the bounding box on the zero mask image
        cv2.fillPoly(bbox_mask, [bbox], color=(255,255,255))
        # cv2.drawContours(bbox_mask,[bbox],-1, (255, 255, 255), 1)

    # This retains only the bounding boxes that overlap with the foreground
    result_mask = cv2.bitwise_and(mask, bbox_mask)
    # _, result_mask = cv2.threshold(result_mask, 127, 255, cv2.THRESH_BINARY)
    result_mask = cv2.cvtColor(result_mask, cv2.COLOR_BGR2GRAY)

    # Finding contours directly from the binary mask
    contours, _ = cv2.findContours(result_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask to draw filled contours
    filled_mask = np.zeros_like(mask, dtype=np.uint8)

    # Draw filled contours on the blank mask
    for contour in contours:
        # Get the bounding rectangle of the contour
        # x, y, w, h = cv2.boundingRect(contour)

        # # Extract the region within the bounding rectangle from the original image
        # region = image[y:y+h, x:x+w]
        # # Calculate the mean color of the extracted region
         
        # mean_color = tuple(np.mean(region, axis=(0, 1)).astype(int))
        # # print(mean_color)
        # mean_color = tuple(map(int, mean_color))
        # print(mean_color)

        # Fill the corresponding region in the mask with the colors from the extracted region
        cv2.fillPoly(mask, [contour], (255,255,255))
        # cv2.drawContours(mask,[contour],-1, (255, 255, 255), 1)


    return mask


bounding_boxes = []
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

def detect_text_in_tile(image, tile_width, tile_height, reader):
    # Initialize a list to store the bounding box coordinates
    bounding_boxes = []
    output_image = np.copy(image)

    # Iterate over each row
    num_rows, num_cols = calculate_num_rows_and_cols(image, tile_width, tile_height)
    for r in range(num_rows):
        # Iterate over each column
        for c in range(num_cols):
            # Calculate the starting coordinates of the tile
            start_x = c * tile_width
            start_y = r * tile_height

            # Extract the tile from the image
            tile = extract_tile(image, start_x, start_y, tile_width, tile_height)

            # Perform text detection on the current tile using the detection model
            # result = reader.readtext(tile, ycenter_ths=0.5, width_ths=0.05, height_ths=0.03, mag_ratio=2.85,
            #                           add_margin=0.2, text_threshold=0.95, workers=6)
            # result = reader.readtext(tile, ycenter_ths=0.5, width_ths=0.05, height_ths=0.03, mag_ratio=2.85,
                                    #   add_margin=0.2, text_threshold=0.2, workers=6)
            result = reader.readtext(tile,text_threshold=0.4)

            # Check if any bounding boxes were returned
            if len(result) > 0:
                # Extract the bounding box coordinates and text from the result
                bounding_boxes_tile = [bbox for bbox, _, _ in result]

                # Map the bounding box coordinates back to the original image coordinates
                for bbox in bounding_boxes_tile:
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

                    mapped_bbox = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    bounding_boxes.append(mapped_bbox)

                    # Draw bounding box on the output image
                    cv2.rectangle(output_image, (int(x1), int(y1)), (int(x3), int(y3)), (0, 0, 255), 2)

    return bounding_boxes, output_image

def main(image_path, tile_width, tile_height):
    # Load the image
    image = load_image(image_path)

    # Initialize EasyOCR reader outside the loop
    reader = easyocr.Reader(['en','te'], gpu=True)  # this needs to run only once to load the model into memory

    # Detect text in tiles
    bounding_boxes, output_image = detect_text_in_tile(image, tile_width, tile_height, reader)
    cv2.imwrite('text_erased_results/result_ca_bounding_box_original_4.png', output_image)
    return bounding_boxes

mask_image_path = '/home/usama/processed_Data_3_july_2024/denoised_data/ca_dana_point/ca_dana_point_4_mask.jpg'  # Replace with your mask image path
mask = cv2.imread(mask_image_path)
image_path = '/home/usama/Converted_jpg_from_tiff_july3_2024/ca_colma.jpg'
image = cv2.imread(image_path)
tile_width = 1024
tile_height = 1024

bounding_boxes= main(image_path, tile_width, tile_height)
print("bounding_boxes : ",bounding_boxes)
mask = draw_bounding_boxes(mask, bounding_boxes,image)



















































