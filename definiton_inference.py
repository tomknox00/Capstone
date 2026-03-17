from inference import get_model
import cv2

def run_detection(image_path, conf=0.5)
    '''
    confidence can be adjusted as needed
    '''
    # Load model
    model = get_model(model_id="plastic-waste-qczkq-j50af/3", api_key="")
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Run inference
    results = model.infer(image, confidence=conf)[0]
    
    # Return the raw predictions list
    return results, image 
