import definition_inference as vision
import find_distance_angle as logic
import supervision as sv
import cv2

# Define the target
target_image = ""

'''
+--------------------------------+
VERY IMPORTANT!!!!!
path to YOUR image must be added 
+--------------------------------+
'''

# 1. Import results from the inference script
results, image = vision.run_detection(target_image)

if results.predictions and len(results.predictions) > 0:
    
    current_width = image.shape[1]

    # 2. Pass those results to the distance/angle functions
    dist = logic.get_distance_to_target(results.predictions)
    angle = logic.get_angle_to_target(results.predictions, image_width=current_width)

    print("-" * 30)
    print(f"PILOT DATA ACQUIRED")
    print(f"Target Distance: {dist:.2f} meters")
    print(f"Target Angle:    {angle:.1f} degrees")
    print("-" * 30)

    # 3. Annotation Logic
    # Passing 'predictions' directly to from_inference is the safest way to avoid KeyErrors
    detections = sv.Detections.from_inference(results)
    
    # Create Annotators
    box_annotator = sv.BoxAnnotator()

    # Create custom labels including the distance we calculated
    labels = [
        f"{p.class_name} {p.confidence:.2f} {dist:.2f}m {angle:.2f}deg"
        for p in results.predictions
    ]

    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections)

    for i, label_text in enumerate(labels):
        # Calculate Y offset so multiple detections stack vertically
        y_coordinate = 50 + (i * 40) 
        
        cv2.putText(
            annotated_frame, 
            label_text, 
            (20, y_coordinate), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.0,           # Font scale slightly reduced for better fit
            (0, 255, 0),   # Color (Green)
            2              # Thickness
        )

    # 4. Show the result
    cv2.imshow("Drone Vision - Plastic Detection", annotated_frame)
    print("Click on the image window and press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("No plastic detected. No navigation data available.")
