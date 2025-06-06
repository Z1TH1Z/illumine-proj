import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_diagonal_rafters(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image could not be read. Check the file path.")

    # Convert to RGB and grayscale
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)

    # Hough Line Transform to find lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=120, maxLineGap=15)

    diagonal_lines = []
    midpoints_x = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if 75 < abs(angle) < 105:  # adjust as needed
                mid_x = (x1 + x2) / 2
                diagonal_lines.append(((x1, y1), (x2, y2)))
                midpoints_x.append(mid_x)
                cv2.line(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Filter duplicate midpoints
        rounded_midpoints = [round(x, 1) for x in midpoints_x]
        unique_midpoints = sorted(list(set(rounded_midpoints)))

        # Compute spacings
        spacings_px = np.diff(unique_midpoints)
        avg_spacing_px = np.mean(spacings_px) if len(spacings_px) > 0 else 0

        # Filter by real-world spacing (12" to 24")
        estimated_spacing_m = 0.4064
        scale_m_per_px = estimated_spacing_m / avg_spacing_px if avg_spacing_px else 0
        spacings_m = spacings_px * scale_m_per_px
        valid = (spacings_m >= 0.3048) & (spacings_m <= 0.6096)
        spacings_px = spacings_px[valid]
        spacings_m = spacings_m[valid]
        avg_spacing_px = np.mean(spacings_px) if len(spacings_px) > 0 else 0
    else:
        scale_m_per_px = 0
        spacings_m = []




    return img_rgb, diagonal_lines, spacings_px, spacings_m, avg_spacing_px, scale_m_per_px

# ---------- MAIN USAGE ------------------
if __name__ == "__main__":
    image_path = "ind_rafter.jpg"  # Change to your image file
    try:
        output_img, lines, spacings_px, spacings_m, avg_px, scale = detect_diagonal_rafters(image_path)

        # Show image with detected lines
        plt.figure(figsize=(12, 8))
        plt.imshow(output_img)
        plt.title("Detected Rafter Lines")
        plt.axis("off")
        plt.show()

        # Print stats
        print(f"✅ Total diagonal rafters detected: {len(lines)}")
        print(f"📏 Average spacing (pixels): {avg_px:.2f}")
        print(f"📐 Estimated pixel-to-meter scale: {scale:.5f} m/px")
        print(f"📌 Example rafter spacings in meters: {spacings_m[:5]}")
        print(f"🎯 Valid spacings between 12\" and 24\": {len(spacings_m)}")

    except Exception as e:
        print(f"Error: {e}")
