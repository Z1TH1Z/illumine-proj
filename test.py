import cv2
import numpy as np

# Define ArUco dictionary and detection parameters (OpenCV 4.7+)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()

# Real-world marker size in cm (width of printed marker)
MARKER_REAL_SIZE_CM = 6.5

# Global variables
pixel_per_cm = None
points = []

# Distance calculator
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Mouse event callback
def click_event(event, x, y, flags, param):
    global points, pixel_per_cm

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

        if len(points) == 2 and pixel_per_cm is not None:
            dist_px = euclidean(points[0], points[1])
            dist_cm = dist_px / pixel_per_cm
            print(f"[INFO] Distance: {dist_cm:.2f} cm")

            cv2.line(param, points[0], points[1], (255, 0, 0), 2)
            cv2.putText(param, f"{dist_cm:.2f} cm", points[1],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            points.clear()

# Open webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("ArUco Ruler")
cv2.setMouseCallback("ArUco Ruler", click_event)

print("📷 Show an ArUco marker (ID 0, 5x5 cm) to calibrate.")
print("🖱️  Click two points to measure real-world distance.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

    output = frame.copy()

    if ids is not None and 0 in ids:
        # Draw marker
        cv2.aruco.drawDetectedMarkers(output, corners, ids)

        # Find marker with ID 0
        index = np.where(ids.flatten() == 0)[0][0]
        marker_corners = corners[index][0]

        # Compute width in pixels
        top_left, top_right = marker_corners[0], marker_corners[1]
        marker_pixel_width = euclidean(top_left, top_right)

        # Calibrate pixels per cm
        pixel_per_cm = marker_pixel_width / MARKER_REAL_SIZE_CM
        cv2.putText(output, f"Calibrated: {pixel_per_cm:.2f} px/cm", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(output, "Show marker ID 0 (5x5cm) to calibrate", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("ArUco Ruler", output)

    key = cv2.waitKey(1)
    if key == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
