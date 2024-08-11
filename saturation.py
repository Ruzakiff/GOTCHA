import cv2
import numpy as np
import sys

def extreme_saturation(image_path):
    # Load image
    image = cv2.imread(image_path)
    
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Extreme saturation increase
    hsv[:,:,1] = 255  # Max out saturation
    
    # Hue shift
    hsv[:,:,0] = (hsv[:,:,0] + 30) % 180  # Shift hue by 30 degrees
    
    # Convert back to BGR
    saturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Increase contrast
    saturated = cv2.addWeighted(saturated, 2, saturated, 0, 0)
    
    return saturated

def main(image_path):
    original = cv2.imread(image_path)
    saturated = extreme_saturation(image_path)
    
    # Combine original and saturated side by side
    height, width = original.shape[:2]
    combined = np.zeros((height, width*2, 3), dtype=np.uint8)
    combined[:, :width] = original
    combined[:, width:] = saturated
    
    cv2.imshow("Original vs Extreme Saturation", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python saturation.py <path_to_image>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    main(image_path)
