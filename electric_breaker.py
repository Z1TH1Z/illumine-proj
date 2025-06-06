import cv2
import pytesseract
import numpy as np
import os

# Set Tesseract path (adjust if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\ILINT111\Desktop\test\Tesseract-OCR\tesseract.exe'

def pre_processing(image, output_path='thresholded.png', scale_factor=2):
    """
    Preprocess an image for OCR with enhanced techniques.
    
    Args:
        image: Input image (numpy array from cv2.imread).
        output_path: Path to save the thresholded image.
        scale_factor: Factor to resize the image for better OCR (default: 2).
    
    Returns:
        threshold_img: Preprocessed binary image, or None if processing fails.
    """
    if image is None:
        print("Error: Input image is None. Check file path or image data.")
        return None

    # Resize image for better OCR of small text
    image_resized = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    # Apply contrast enhancement
    gray_image = cv2.equalizeHist(gray_image)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)

    # Adaptive thresholding for better contrast
    threshold_img = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 2
    )

    # Dilate to connect broken characters
    kernel = np.ones((2, 2), np.uint8)
    threshold_img = cv2.dilate(threshold_img, kernel, iterations=1)

    # Save the thresholded image for inspection
    try:
        cv2.imwrite(output_path, threshold_img)
        print(f"Thresholded image saved as {output_path}")
    except Exception as e:
        print(f"Error saving image: {e}")
        return threshold_img

    # Display the image (skip in non-interactive environments)
    if os.environ.get('DISPLAY'):
        cv2.imshow('Threshold Image', threshold_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Display not available, skipping image display.")

    return threshold_img, image_resized

def process_lcd(image, output_path='lcd_thresholded.png'):
    """
    Process the LCD region separately for better digit extraction.
    
    Args:
        image: Input image (numpy array).
        output_path: Path to save the thresholded LCD image.
    
    Returns:
        lcd_text: Extracted text from the LCD.
    """
    # Crop the LCD region (adjust coordinates based on the image)
    lcd_region = image[100:150, 50:250]  # Approximate coordinates for "014 22790"
    
    # Convert to grayscale and enhance contrast
    lcd_gray = cv2.cvtColor(lcd_region, cv2.COLOR_BGR2GRAY)
    lcd_contrast = cv2.equalizeHist(lcd_gray)

    # Apply adaptive thresholding
    lcd_threshold = cv2.adaptiveThreshold(
        lcd_contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Save the LCD thresholded image for inspection
    cv2.imwrite(output_path, lcd_threshold)

    # OCR for LCD (digits only)
    lcd_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
    lcd_text = pytesseract.image_to_string(lcd_threshold, config=lcd_config)
    return lcd_text.strip()

# Load and process the image
image_path = 'elec_break.png'
image = cv2.imread(image_path)

if image is not None:
    # Process the entire image
    thresholded_image, resized_image = pre_processing(image, scale_factor=2)
    if thresholded_image is not None:
        try:
            # OCR for the entire image
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-./()*'
            text = pytesseract.image_to_string(thresholded_image, config=custom_config)
            print("Extracted Text (Full Image):", text)

            # Process the LCD separately
            lcd_text = process_lcd(resized_image)
            print("LCD Text:", lcd_text)
        except pytesseract.TesseractNotFoundError:
            print("Error: Tesseract is not installed or not in PATH. Please install Tesseract or set pytesseract.pytesseract.tesseract_cmd.")
        except Exception as e:
            print(f"Error during OCR: {e}")
else:
    print(f"Error: Could not load image at {image_path}")