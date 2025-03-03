import os
import io
import numpy as np
from PIL import Image

def is_valid_image(img_path):
    """
    Validate if an image can be properly loaded and is of usable quality
    
    Args:
        img_path: Path to the image file
        
    Returns:
        bool: True if image is valid, False otherwise
    """
    try:
        with Image.open(img_path) as img:
            # Check if image is valid
            img.verify()
            
            # Open image again to check if it can be loaded properly
            img = Image.open(img_path)
            
            # Convert to RGB (some images might be grayscale or RGBA)
            img = img.convert('RGB')
            
            # Check minimum resolution (80x80)
            width, height = img.size
            if width < 80 or height < 80:
                print(f"Image too small: {img_path}, size: {width}x{height}")
                return False
                
            # # Check JPEG quality if it's a JPEG
            # if img_path.lower().endswith('.jpg') or img_path.lower().endswith('.jpeg'):
                # try:
                #     img.save('temp.jpg', quality=20)
                #     test_img = Image.open('temp.jpg')
                #     orig_pixels = np.array(img)
                #     test_pixels = np.array(test_img)
                #     quality_ratio = np.mean(np.abs(orig_pixels - test_pixels)) / 255.0
                #     if quality_ratio > 0.2:  # Threshold for quality
                #         print(f"Image quality too low: {img_path}")
                #         return False
                # finally:
                #     if os.path.exists('temp.jpg'):
                #         os.remove('temp.jpg')
                # Create a buffer to store the image
                # buffer = io.BytesIO()
                # # Save the image to the buffer with quality 20
                # img.save(buffer, format="JPEG", quality=20)
                # # Get the size of the compressed image
                # size_at_20 = buffer.tell()
                
                # # Reset the buffer
                # buffer.seek(0)
                # buffer.truncate(0)
                
                # # Save the original image to the buffer
                # img.save(buffer, format="JPEG", quality=95)
                # # Get the size of the original image
                # original_size = buffer.tell()
            
                # # Compare sizes
                # quality_ratio = size_at_20 / original_size

                # if quality_ratio > 0.2:  # Threshold for quality
                #     print(f"Image quality too low: {img_path}")
                #     return False
            
            return True
    except Exception as e:
        print(f"Invalid image {img_path}: {str(e)}")
        return False