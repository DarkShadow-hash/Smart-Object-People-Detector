import os
import time
import cv2
from ultralytics import YOLO

# Configuration and paths (adjust paths as needed). 
# This is the path to the custom model weights (best.pt) trained on Google Colab.
PATH_TO_CUSTOM_MODEL = './train2_results/weights/best.pt' 
# Source 0 for the webcam, or 'path/to/video.mp4' for a fileSource 0 is the default for the local webcam. Change this to a file path (e.g., 'test.mp4') 
# to run detection on a pre-recorded video.
VIDEO_SOURCE = 0 
# The Confidence Threshold is set to the standard 0.50 (50% sure) for a balanced output.
CONFIDENCE_THRESHOLD = 0.50 
# Dictionary to store the object's unique ID and its starting timestamp
OBJECT_START_TIME = {} 

def track_and_time_objects(model_path, source, conf_thresh):
    """
    Loads the custom-trained YOLOv8 model, performs object detection on a video stream
    (webcam or file), and implements a simple timer to measure how long each 
    detected object stays in the field of view.
    
    This function demonstrates object tracking and temporal analysis as required 
    for the advanced project component.

    Args:
        model_path (str): Path to the trained model weight file (best.pt).
        source (int/str): The video source. '0' is standard for the local webcam.
        conf_thresh (float): The minimum confidence threshold for a detection to be displayed.
    
    Returns:
        None: Opens an OpenCV window and displays the real-time detection.
        
    Raises:
        Exception: If the YOLO model file cannot be found (e.g., if best.pt is missing).
    """
    print(f"Loading custom model: {model_path}...")
    try:
        # Attempts to load our custom trained model
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Use OpenCV to handle the video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return

    print("Launching real-time detection (Press 'q' to quit)...")

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        current_time = time.time()
        
        # 1. Prediction using the custom model
        # We process the frame and apply the chosen confidence threshold
        results = model(frame, conf=conf_thresh, verbose=False)
        
        # 2. Process Detections for Timer Logic
        # Extract the box data (coordinates, confidence, and class ID)
        detections = results[0].boxes.data.tolist()
        
        if detections:
            for detection in detections:
                x1, y1, x2, y2, conf, cls = detection
                
                # Simple tracking ID creation :
                # We create a unique ID based on the object's class ID and its approximate 
                # horizontal position (x1/50). This simple method is used for tracking 
                # since the object's box should not change rapidly.
                object_id = f"{int(cls)}-{int(x1/50)}" 
                
                # Timer logic :
                if object_id not in OBJECT_START_TIME:
                    # If this is the first frame we see this object ID, we start the timer
                    OBJECT_START_TIME[object_id] = current_time
                
                # Calculate the elapsed time
                time_elapsed = current_time - OBJECT_START_TIME[object_id]
                time_display = f"({model.names[int(cls)]}) {int(time_elapsed)}s"
                
                # Drawing output using OpenCV :
                # Draw the bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Display the label and the timer value above the box
                cv2.putText(frame, time_display, 
                            (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 3. Display frame and exit control
        # cv2.imshow displays the video window
        cv2.imshow("Detection with Timer (Jour 4)", frame)
        
        # cv2.waitKey(1) is necessary for the window to refresh (waits 1ms).
        # We check for the 'q' key press to close the application cleanly.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 4. Release resources (clean up)
    # Releases the webcam and closes all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # We call the main function here. The execution must happen within the activated virtual environment 
    # that contains the necessary libraries listed in requirements.txt.
    track_and_time_objects(PATH_TO_CUSTOM_MODEL, VIDEO_SOURCE, CONFIDENCE_THRESHOLD)
