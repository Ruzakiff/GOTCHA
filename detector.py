import cv2
import numpy as np
import sys
from skimage.feature import local_binary_pattern
from scipy.fftpack import dct

def extreme_saturation(image):
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

def compute_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    return lbp

def color_quantization(image, n_colors=8):
    Z = image.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(Z, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape((image.shape))

def frequency_analysis(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dct_result = dct(dct(gray.T, norm='ortho').T, norm='ortho')
    return np.sum(np.abs(dct_result[:10, :10])) / (10 * 10)

def analyze_global_features(original, saturated):
    # Analyze color distribution
    color_diff = np.mean(np.abs(original.astype(float) - saturated.astype(float)))
    
    # Analyze edge consistency
    edges_original = cv2.Canny(original, 100, 200)
    edges_saturated = cv2.Canny(saturated, 100, 200)
    edge_diff = np.mean(cv2.absdiff(edges_original, edges_saturated))
    
    # Analyze frequency domain
    freq_original = frequency_analysis(original)
    freq_saturated = frequency_analysis(saturated)
    freq_diff = abs(freq_original - freq_saturated)
    
    # Combine global features
    global_score = color_diff * edge_diff * freq_diff
    return global_score

def detect_local_artifacts(original, saturated):
    # LBP analysis
    lbp_original = compute_lbp(original)
    lbp_saturated = compute_lbp(saturated)
    lbp_diff = np.abs(lbp_original - lbp_saturated)
    
    # Color quantization analysis
    quant_original = color_quantization(original)
    quant_saturated = color_quantization(saturated)
    quant_diff = np.sum(np.abs(quant_original.astype(int) - quant_saturated.astype(int)), axis=2)
    
    # Edge coherence
    edges_original = cv2.Canny(original, 100, 200)
    edges_saturated = cv2.Canny(saturated, 100, 200)
    edge_diff = cv2.absdiff(edges_original, edges_saturated)
    
    # Frequency analysis
    freq_original = frequency_analysis(original)
    freq_saturated = frequency_analysis(saturated)
    freq_diff = abs(freq_original - freq_saturated)
    
    # Combine all differences
    combined_diff = lbp_diff * quant_diff * edge_diff * freq_diff
    
    # Threshold the combined difference
    _, thresh = cv2.threshold(combined_diff, 1000, 255, cv2.THRESH_BINARY)
    thresh = thresh.astype(np.uint8)
    
    # Apply morphological operations
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area and shape
    min_area = 100
    max_area = 10000
    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.2:  # Filter out very irregular shapes
                filtered_contours.append(cnt)
    
    local_score = len(filtered_contours) / (original.shape[0] * original.shape[1])
    return filtered_contours, local_score

def classify_image(global_score, local_score):
    # Thresholds for classification
    global_threshold = 1000  # Adjust based on experimentation
    local_threshold = 0.0001  # Adjust based on experimentation
    
    if global_score > global_threshold and local_score < local_threshold:
        return "AI-generated", (global_score + local_score) / 2
    else:
        return "Real", (global_score + local_score) / 2

def highlight_artifacts(image, contours):
    # Create a copy of the image
    highlighted = image.copy()
    
    # Draw contours on the image
    cv2.drawContours(highlighted, contours, -1, (0, 255, 0), 2)
    
    return highlighted

def main(image_path):
    # Read the original image
    original = cv2.imread(image_path)
    
    # Apply extreme saturation
    saturated = extreme_saturation(original)
    
    # Analyze global features
    global_score = analyze_global_features(original, saturated)
    
    # Detect local artifacts
    contours, local_score = detect_local_artifacts(original, saturated)
    
    # Classify the image
    classification, confidence = classify_image(global_score, local_score)
    
    # Highlight artifacts on both original and saturated images
    highlighted_original = highlight_artifacts(original, contours)
    highlighted_saturated = highlight_artifacts(saturated, contours)
    
    # Combine images for display
    top_row = np.hstack((original, saturated))
    bottom_row = np.hstack((highlighted_original, highlighted_saturated))
    combined = np.vstack((top_row, bottom_row))
    
    # Display results
    cv2.imshow("Artifact Detection", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Print detection results
    print(f"Classification: {classification}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Detected {len(contours)} potential artifacts.")
    print(f"Global Score: {global_score:.4f}")
    print(f"Local Score: {local_score:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detector.py <path_to_image>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    main(image_path)
    sys.exit(1)
    
