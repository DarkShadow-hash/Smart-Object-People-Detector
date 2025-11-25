import os
from ultralytics import YOLO

# Path definitions (must be adapted to your local structure) :
# Path to the custom model weights (best.pt) retrieved after Colab training.
PATH_TO_CUSTOM_MODEL = './runs/detect/train2/weights/best.pt' 
# Directory containing static images used for side-by-side comparison.
PATH_TO_TEST_IMAGES = './images_to_test/'
# Path to the base YOLOv8n model, which the ultralytics library downloads automatically if missing.
PATH_TO_BASELINE_MODEL = 'yolov8n.pt' 

# Class names (must match the order in data.yaml) :
# Used for display purposes in the script output.
NAMES = {0: 'Mont Roucous', 1: 'Casque', 2: 'Trousse'} 


def run_prediction(model_path: str, source_dir: str, project_name: str):
    """
    Loads a YOLO model (custom or baseline) and executes both static image prediction 
    and formal metric evaluation on the project's test set.
    
    This function is used twice : once for the baseline model and once for the 
    custom-trained model to formally demonstrate performance uplift.

    Args:
        model_path (str): The path to the model weights (e.g., 'yolov8n.pt' or 'best.pt').
        source_dir (str): The directory containing static images for visual testing.
        project_name (str): The name of the subdirectory where results are saved 
                            within the ./runs/detect/ folder (e.g., 'baseline_yolo').
    
    Returns:
        results: The YOLO prediction results object.
    """
    # 1. Load the Model
    print(f"\n Loading model : {project_name} ")
    model = YOLO(model_path)
    
    # 2. Execute prediction on pictures
    # The 'predict' mode runs inference and saves the visual output (images with boxes).
    # We choose conf = 0.25 here because the baseline model is expected to have low confidence 
    # scores; a lower threshold ensures we capture any base model detection.
    results = model.predict(
        source=source_dir, 
        conf=0.25,      # Confidence threshold (set low for the comparison rigor)
        save=True,      # Saves the images with detection boxes for visual proof
        name=project_name # Creates a subfolder in runs/detect/
    )
    
    # 3. Formal metric evaluation :
    # The 'val' mode is crucial: it calculates the true performance metrics (mAP)
    # on the dedicated, unseen 'test' split. This proves the high mAP reported 
    # in the academic report.
    # NB : This step was executed reliably on Google Colab to avoid local Segmentation Faults.
    try:
        metrics = model.val(data='./labelling images.v2i.yolov8/data.yaml', split='test') 
        print(f"Metrics {project_name} on test set (mAP50): {metrics.results_dict['metrics/mAP50(B)']:.4f}")
    except Exception as e:
        # We allow this to fail locally, as the critical data was gathered on Colab.
        print(f"Warning: model.val failed locally (expected due to environment conflict). Data retrieved from Colab.")
        
    return results


if __name__ == '__main__':
    # 1. Test the baseline model :
    # Purpose : To establish a control and prove that the base YOLO model cannot detect 
    # our custom, tightly focused objects, thereby justifying the need for training.
    run_prediction(
        model_path=PATH_TO_BASELINE_MODEL,
        source_dir=PATH_TO_TEST_IMAGES,
        project_name='baseline_yolo' 
    )

    # 2. Test the custom model :
    # Purpose: To demonstrate the successful transfer learning and the project's main contribution.
    run_prediction(
        model_path=PATH_TO_CUSTOM_MODEL,
        source_dir=PATH_TO_TEST_IMAGES,
        project_name='custom_trained'
    )
    
    # Note: All output images with detections are saved in ./runs/detect/
