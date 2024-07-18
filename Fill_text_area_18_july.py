

import cv2
import numpy as np
import os

def merge_original_mask_with_bounding_mask(original_mask_path, bounding_box_mask_path):
    # Read the images
    original_mask = cv2.imread(original_mask_path)
    # _,original_mask = cv2.threshold(original_mask,2,255,cv2.THRESH_BINARY)
    bounding_box_mask = cv2.imread(bounding_box_mask_path)
    _,bounding_box_mask = cv2.threshold(bounding_box_mask,2,255,cv2.THRESH_BINARY)

    # Resize the bounding box mask to match the dimensions of the original mask
    bounding_box_mask = cv2.resize(bounding_box_mask, (original_mask.shape[1], original_mask.shape[0]))

    # Perform bitwise AND operation to combine the masks
    result_bounding_mask = cv2.bitwise_and(bounding_box_mask, original_mask)
    result_bounding_mask_gray = cv2.cvtColor(result_bounding_mask, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(result_bounding_mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask to draw filled contours
    filled_bounding_box_mask = np.zeros_like(original_mask, dtype=np.uint8)

    for contour in contours:
        cv2.fillPoly(filled_bounding_box_mask, [contour], (255, 255, 255))

    # Merge the filled bounding box mask with the original mask
    merged_mask = cv2.bitwise_or(filled_bounding_box_mask, original_mask)

    return merged_mask

# Directory paths
mask_directory = '/home/usama/Denoised_mask_results_3_july_2024/ca_millbrae_enhanced1/'
bounding_box_mask_path = 'text_masks_results_july_13_2024/ca_millbrae_enhanced1text_masktext_mask.png'
output_directory = 'Merged_results_18_july_2024/'

os.makedirs(output_directory, exist_ok=True)

for original_mask_file in os.listdir(mask_directory):
    if original_mask_file.endswith(('.png', '.jpg')):
        original_mask_path = os.path.join(mask_directory, original_mask_file)
        base_name = os.path.splitext(os.path.basename(original_mask_path))[0]  # Get the base name without extension
        output_file_path = os.path.join(output_directory, f"{base_name}_merged.png")

        # Merge the original mask with the bounding box mask
        merged_mask = merge_original_mask_with_bounding_mask(original_mask_path, bounding_box_mask_path)

        # Save the merged mask
        cv2.imwrite(output_file_path, merged_mask)



# cv2.imwrite("result_bounding_intersected_mask.png",filled_bounding_box_mask)