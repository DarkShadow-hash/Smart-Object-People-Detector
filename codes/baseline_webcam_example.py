import cv2
from ultralytics import YOLO
import time

# Load a base model pre-trained on the COCO dataset. 
# This is NOT the custom model we trained : this is used as a baseline reference.
model = YOLO("yolov8m.pt") 

# Use OpenCV's VideoCapture object to open the local webcam (source 0).
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access your webcam.")
    # Exits the program if the webcam cannot be initialized.
    exit()

def run_baseline_webcam_detection():
    """
    Runs real-time object detection using a pre-trained YOLOv8m model on the webcam feed.
    
    This script is intended as a baseline benchmark to demonstrate the difference 
    between general-purpose detection (YOLOv8m) and our custom-trained solution 
    (which uses the custom model and the Object Timer).

    Args:
        None (Uses global model and webcam source).
    
    Returns:
        None: Displays the video feed in an OpenCV window until the user quits.
    """
    while True:
        # Read a new frame from the webcam feed
        ret, frame = cap.read()
        if not ret:
            # Breaks the loop if reading the frame fails
            break

        t0 = time.time()
        # Perform detection on the current frame.
        # conf=0.5 and iou=0.5 are standard thresholds for general detection.
        results = model(frame, conf=0.5, iou=0.5)
        
        # results[0].plot() generates a new image with the detection boxes and labels drawn.
        annotated = results[0].plot()
        
        # Simple calculation for Frames Per Second (FPS)
        fps = 1.0 / max(time.time() - t0, 1e-6)

        # Display the calculated FPS on the annotated frame
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the output window
        cv2.imshow("Webcam Detection - Baseline", annotated)

        # Exit condition: close the window if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_baseline_webcam_detection()
