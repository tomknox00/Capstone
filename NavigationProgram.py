"""
+---------------------------------------------------+
| Insert titile block text here
|
|
|
+---------------------------------------------------+
"""

#Check if garbage has been detected by the garbage detection program
def get_is_garbage_detected():
	pass #[IMPLEMENT CODE]: Replace with function call to garbage detection program
	return False

#Get the angle to target in degrees from garbage detection program
def get_angle_to_target():
	pass #[IMPLEMENT CODE]: Replace with function call to garbage detection program
	return 0

#Get the distance to target from garbage detection program in meters
def get_distance_to_target():
	pass #[IMPLEMENT CODE]: Replace with function call to garbage detection program
	return 0

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

if get_battery_percentage <= 50: #If battery is low, return to port
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
