"""
+---------------------------------------------------+
| Code for steering the drone towards garabge       |
| Replace unimplemented functions with calls        |
| to the garabge detection and motor drive.         |
|                                                   |
+---------------------------------------------------+
"""

from inference import get_model
import cv2

# --- CONFIGURATION ---
CENTER_X = 320         # Middle of the 640px frame
FOV_H = 62.4           # Horizontal Field of View (degrees)
DIST_K = 140986        # Constant for Pi camera module 2 @ 640px

"""
+--------------------------------------------------------------------------------------------------------------------+
| What is the DIST_K Constant?				        																 |
| It is the Apparent Focal Constant. It is a "shortcut" number that combines three physical properties into one:	 |
| Focal Length (f): magnification of lens (3.04 mm).  																 |
| Sensor Scale: The conversion of physical millimeters on the silicon chip into pixels (based on 1.12 µm pixel size).|
| Reference Object Height: real-world size of the plastic object (in our case 200 mm bottle).						 |
|																													 |
| DIST_K=(f/Hsensor *​ Vres) * Htarget_mm																			 |
| DIST_K=(Focal pixels)×(Real height in mm)																			 |
|																													 |​
+--------------------------------------------------------------------------------------------------------------------+
"""

# Initialize Model
model = get_model(model_id="plastic-waste-qczkq-j50af/3", api_key="")

#--------------------------------------------Functions-------------------------------------------------

#Check if garbage has been detected by the garbage detection program
def get_is_garbage_detected():
	pass #[IMPLEMENT CODE]: Replace with function call to garbage detection program
	return False

#Get the angle to target in degrees from garbage detection program
def get_angle_to_target(predictions, image_width=None):
	
	"""
	+---------------------------------------------------+
	| Calculates the angle to target by determining the |
	| center dynamically from the current image width.  |     
	| Replace unimplemented functions with calls to the |
	| garabge detection and motor drive.                |
	|                                                   |
	+---------------------------------------------------+
	"""
	
    if not predictions:
        return 0
    
    # If width isn't passed, try to extract it from the prediction metadata
    # This ensures the center is always relative to the current frame
    if image_width is None:
        # In the Roboflow SDK, predictions often carry the parent image width
        # If your SDK version doesn't support this, we default to the standard width
        image_width = getattr(predictions[0], 'image_width', 320)

    # Calculate the dynamic center
    center_x = image_width / 2
    
    # Target x is the center of the bounding box
    target_x = predictions[0].x
    
    # Precise angle calculation
    angle = ((target_x - center_x) / image_width) * FOV_H
    
    return angle

#Get the distance to target from garbage detection program in meters
def get_distance_to_target(predictions):

	"""
	+--------------------------------------------------------------------------------+
    | Calculates the distance to the detected plastic using the Pinhole Camera Model.|
    |																				 |
    | Logic: Uses the 'Distance Constant' (DIST_K), which represents the product     |
    | of the camera's focal length in pixels and the known real-world height of      |
    | the target (e.g., 200mm for a standard bottle).								 |
    | 																				 |
    | Formula: Distance = (Focal_Pixels * Real_Height_mm) / Vertical_Pixel_Height    |
    | 																				 |
    | Returns: Distance in meters.													 |
	+--------------------------------------------------------------------------------+
    """
	
	if not predictions: return 0
    h_yolo = predictions[0].height
    # Returns meters (e.g., 0.51)
    return (DIST_K / h_yolo) / 1000

#Get the percentage of power remaining in the battery
def get_battery_percentage():
	pass #[IMPLEMENT CODE]: Replace with function call to battery reading program
	return 100

#Controls the speeds of both motors
def set_motor_drives(left_motor, right_motor):
	if abs(left_motor) <= 1 and abs(right_motor) <=1:
		pass
		#[IMPLEMENT CODE]: Set the speed of the left motor, 1 = max speed forward, -1 = max speed reverse
		#[IMPLEMENT CODE]: Set the speed of the right motor, 1 = max speed forward, -1 = max speed reverse
	else
		raise ValueError("Motor drives must be bettween 1 and -1") 


"""
+-----------------------------+
| Main code block starts here |
+-----------------------------+
"""

'''
if get_battery_percentage() <= 50: #If battery is low, return to port
	pass #[IMPLEMENT CODE]: Order the robot to return to base

elif get_is_garbage_detected(): #If floating garabge is detected
	while get_is_garbage_detected(): #If the target is still in sight
		angle_to_current_target = get_angle_to_target() #Store angle to current target

		if angle_to_current_target == 0:
			set_motor_drives(1, 1) #Move forward

		elif angle_to_current_target > 10
			set_motor_drives(-1, 1) #Turn left on the spot

		elif angle_to_current_target < -10
			set_motor_drives(1, -1) #Turn right on the spot

		elif angle_to_current_target <= 10 and angle_to__current_target > 0
			set_motor_drives(0.5, 1) #Turn left slightly while moving forward

		elif angle_to_current_target >= -10 and angle_to__current_target < 0
			set_motor_drives(1, 0.5) #Turn right slightly while moving forward

	#[IMPLEMENT CODE]: Move robot forward one meter to ensure collection
else:

	pass #[IMPLEMENT CODE]: Continue to search for garbage
'''

