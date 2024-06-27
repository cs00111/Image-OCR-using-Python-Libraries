import pytesseract
import cv2
import os
import numpy as np

def clear_output_file(output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("")  
# Overwrite file with empty content

# Load the invoice image
image = cv2.imread('.venv/new.jpg')

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Write grayscale image for reference
cv2.imwrite('.venv/gray_scale.jpg', gray)

# Threshold to binary image
thresh, im_bw = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
cv2.imwrite('.venv/bw.jpg', im_bw)

# Erode the binary image
kernel1 = np.ones((1, 1), np.uint8)
eroded = cv2.erode(im_bw, kernel1, iterations=5)
cv2.imwrite('.venv/eroded.jpg', eroded)

# Blur the eroded image
blur = cv2.GaussianBlur(eroded, (7, 7), 5)
cv2.imwrite('.venv/blur.jpg', blur)

# Modify the image threshold and invert it to find contours
threshed = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
cv2.imwrite('.venv/thresh.jpg', threshed)

# Create a kernel for dilating contours
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
cv2.imwrite('.venv/kernel.jpg', kernel)

# Dilate the image to merge adjacent contours
dilated = cv2.dilate(threshed, kernel, iterations=6)
cv2.imwrite('.venv/dilated.jpg', dilated)

# Find and sort contours by their y-coordinate
cnts = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[1])

# Function to calculate average confidence score
def calculate_conf(data):
    total_confidence = 0.0
    count = 0
    for i in range(len(data['conf'])):
        try:
            confidence = float(data['conf'][i])
            if confidence != -1:
                total_confidence += confidence
                count += 1
        except ValueError:
            continue  # Skip if conversion to float fails
    if count > 0:
        return total_confidence / count
    else:
        return 0.0  # Return 0 if no valid confidence scores found

# Process each contour (ROI)
sum_confidence = 0.0
valid_count = 0

output_file = '.venv/output_combined.txt'

clear_output_file(output_file)

# Open file in append mode
with open(output_file, 'a', encoding='utf-8') as f:
    for idx, c in enumerate(cnts):
        x, y, w, h = cv2.boundingRect(c)
        if h > 38 and y > 20:  # Filter based on height and position (adjust as needed)
            roi = image[y:y+h, x:x+w]
            cv2.rectangle(image, (x, y), (x+w, y+h), (36, 255, 12), 2)
            ocr_data = pytesseract.image_to_data(roi, output_type=pytesseract.Output.DICT)
            conf = calculate_conf(ocr_data)
            sum_confidence += conf
            valid_count += 1
            # Get OCR text
            roi_text = pytesseract.image_to_string(roi)
            
            # Write ROI text to the output file
            f.write(roi_text + '\n')
            
print(ocr_data)

print(f'Overall confidence Score: {sum_confidence/valid_count : .3f}')
# Save the image with bounding boxes
cv2.imwrite('.venv/contours.jpg', image)

print("OCR results combined and saved to:", output_file)
