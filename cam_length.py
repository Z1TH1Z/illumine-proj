import cv2
import numpy as np


points = []
pixel_per_cm = None
ref_object_width_cm = 20.3  


def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def click_event(event, x, y, flags, param):
    global points, pixel_per_cm

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

        if len(points) == 2:
            if pixel_per_cm is None:
              
                pixel_per_cm = euclidean(points[0], points[1]) / ref_object_width_cm
                print(f"[Calibration] Pixels per cm: {pixel_per_cm:.2f}")
                cv2.line(param, points[0], points[1], (0, 255, 0), 2)
                cv2.putText(param, "Calibrated", points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                
                pixel_dist = euclidean(points[0], points[1])
                cm_dist = pixel_dist / pixel_per_cm
                print(f"[Measurement] Distance = {cm_dist:.2f} cm")
                cv2.line(param, points[0], points[1], (255, 0, 0), 2)
                cv2.putText(param, f"{cm_dist:.2f} cm", points[1], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            points.clear()

cap = cv2.VideoCapture(0)


cv2.namedWindow("Live Measurement")
cv2.setMouseCallback("Live Measurement", click_event)

print("Click two points on a known-width object (e.g., A4 paper) to calibrate.")
print("Then click two points to measure actual distance.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    display_frame = frame.copy()


    cv2.imshow("Live Measurement", display_frame)
    cv2.setMouseCallback("Live Measurement", click_event, display_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
