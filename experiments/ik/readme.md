Here is the inverse kinematics module that uses the ikpy library
for generating the set of joint angles permissible for generating a
certain 6D end-effector pose.

# Instructions

A dataset contains the set of positions, the natural language instruction, and the folder name. You will need to define all of these, or set the language or folder to a placeholder and change it later. You will have to update the file paths for dataset_dir and the seal urdf file. dataset_dir leads to the parent file where all the example folder will be in. 

If you want to run the ik_test.ipynb, you will need to pip install ipympl or conda install -c conda-forge ipympl.

## What to Change for each sample

Coordinates: change the set of coordinates

known_coordinates: this is the set of x,y,z coordinates in meters. You can directly put in the coordinates where known_coordinates["blah_blah"] is.

known_orientation: this is the orientation of gripper as yaw, pitch, roll, in radians. The known_orientation object is initially defined in degrees, and later changed to radians, so that the end_effector_angles is in radians when the file is made. I'd recommend putting any orientations you need into known_orientation as degrees, then using it in coordinates, as working with radians is annoying.

gripper_state: this is how much the gripper is gripping. Minimum angle achieved is at 0.2, maximum angle at 0.8. The other values are the smallest object width that the value can grab. I'd recommend setting the next smallest option to the object you are picking up. 
WARNING: the gripper currently stalls out when gripping hard objects. When it stalls out, it will continue trying to get to the desired angle, regardless of any new angles sent to it. This results in the robot not dropping items sometimes. The robot will still pick up items just fine.

Instruction: this is the natural language instruction that you would give to the VLA ie 'pick up the yellow block and move it to the right'. The language_instruction.txt file will be created. You can put in a placeholder and copy/paste that when creating a new sample, then edit it later if you want. 

Folder: this is the name of the folder the sample will be placed in, ie 'block_left_mid_to_right_mid'. Once again, you can put in a placeholder and copy/paste that when creating a new sample, then edit it later if you want. 

# What to look out for

When in doubt, turn off power and restart the Arduino via the button on the board.

If the arm is moving weirdly or not responding or moving very slowly, turn off the power briefly.

If the gripper is permanently closed, it is because the servo is stalling. Turning off the power and restarting the Arduino should fix this issue. I have a potential untested fix in the Arduino code that is commented out if you want to try it out. Otherwise, try setting the gripper to slightly large values until it grips without stalling.
