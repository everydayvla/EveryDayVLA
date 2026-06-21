import math
from math import pi
import numpy as np


deg2rad = pi / 180
rad2deg = 180 / pi
# print(-0.05 * rad2deg)

smooth_step= 0.015      # step

modify_coords = lambda coords, x, y, z: [x, y, z, *coords[3:]]

known_coordinates = {
    "center_mid_grounded": [0.18,0.0,0.02],
    "center_mid_raised": [0.18,0,0.1],
    "center_close_grounded": [0.1,0,0.02],
    "center_close_raised": [0.1, 0, 0.1],
    "center_far_grounded": [0.30, 0, 0],
    "center_far_grounded_side": [0.30, 0, 0.02],
    "left_close_grounded":[0.07, 0.07, 0.02,],
    "left_close_raised": [0.07, 0.07, 0.15],
    "left_mid_grounded": [0.14, 0.14, 0.02],
    "left_mid_raised": [0.14, 0.14, 0.1],
    "right_close_grounded":[0.07, -0.07, 0.02],
    "right_close_raised": [0.07, -0.07, 0.15],
    "right_mid_grounded": [0.14, -0.14, 0.02],
    "right_mid_raised": [0.14, -0.14, 0.1],
    
}

known_orientations = {
    "vertical_forward": [0, 180, 0],
    "vertical_left": [90, 180, 0],
    "vertical_right": [-90, 180, 0],
    "vertical_slight_left": [45, 180, 0],
    "vertical_slight_right": [-45, 180, 0],
    "side_forward": [0, 90, 0],
    "side_left": [90, 90, 0],
    "side_right": [-90, 90, 0],
    "side_slight_left": [45, 90, 0],
    "side_slight_right": [-45, 90, 0],
}
for n in known_orientations:
    known_orientations[n] = [x*deg2rad for x in known_orientations[n]]

gripper_state = {
    "hold":[0],
    "1cm": [0.25],
    "3cm": [0.3],
    "5cm": [0.35],
    "release":[1]
}

object_sizes = {
    "ball": gripper_state["1cm"],
}

height_adjust = lambda start: 0.02 if np.linalg.norm(start[:2]) > 0.27 else (0.01 if np.linalg.norm(start[:2]) > 0.21 else 0.00)

set_coords = lambda start, end, orientation1, orientation2, grip:[
    start[:2] + [start[2] + 0.1] + orientation1 + gripper_state["release"],
    start + orientation1 + gripper_state["release"], 
    start + orientation1 + grip,
    start[:2] + [start[2] + 0.1] + orientation1 + grip,
    end[:2] + [end[2] + 0.1] + orientation2 + grip,
    end + orientation2 + grip,
    end + orientation2 + [grip[0]+0.3],
    ]
set_coords_cup = lambda start, end, orientation1, orientation2, grip:[
    start[:2] + [start[2] + 0.15] + orientation1 + gripper_state["release"],
    start[:2] + [start[2] + height_adjust(start[:2])] + orientation1 + gripper_state["release"], 
    start[:2] + [start[2] + height_adjust(start[:2])] + orientation1 + grip,
    start[:2] + [max(start[2] + 0.15, end[2] + height_adjust(end[:2]))] + orientation1 + grip,
    end[:2] + [end[2] + height_adjust(end[:2])] + orientation2 + grip,
    end[:2] + [end[2] + height_adjust(end[:2])] + orientation2 + gripper_state["release"], 
    ]

set_coords_rotation = lambda start, end, orientation1, orientation2, grip:[
    start[:2] + [start[2] + 0.08] + orientation1 + gripper_state["release"],
    start + orientation1 + gripper_state["release"], 
    start + orientation1 + grip,
    # start[:2] + [start[2] + 0.15] + orientation1 + grip,
    end + orientation2 + grip,
    end + orientation2 + gripper_state["release"], 
    ]

'''
Dataset notes:
object sizes:
    creeper: 3cm
    block: 3cm
    
'''

place_block = lambda start, end, orientation1, orientation2, grip:[
    start[:2] + [max(start[2] + 0.1, end[2] + 0.12)] + orientation1 + gripper_state["release"],
    start + orientation1 + gripper_state["release"], 
    start + orientation1 + grip,
    start[:2] + [start[2] + 0.1] + orientation1 + grip,
    end[:2] + [max(start[2] + 0.1, end[2] + 0.12)] + orientation2 + grip,
    end + orientation2 + grip,
    end + orientation2 + gripper_state["release"],
    end[:2] + [max(start[2] + 0.1, end[2] + 0.12)] + orientation2 + gripper_state["release"],
    ]
rotate_in_place = lambda start, orientation1, orientation2, grip:[
    
]

pour_water = lambda start, end, orientation1, orientation2, grip:[
    [start[0] - 0.07*math.cos(math.atan2(start[1], start[0])),start[1] - 0.07*math.sin(math.atan2(start[1], start[0])), start[2]] + [math.atan2(start[1], start[0]), pi/2, 0] + [1],
    start + [math.atan2(start[1], start[0]), pi/2, 0] + [1],
    start + [math.atan2(start[1], start[0]), pi/2, 0] + grip,
    start[0:2] + [start[2] + 0.15]  + [math.atan2(start[1], start[0]), pi/2, 0] + grip,
    end + orientation1 + grip,
    end + orientation2 + grip,
    end + orientation2 + grip,
    end + orientation1 + grip,
    start[0:2] + [start[2] + 0.1]  + [math.atan2(start[1], start[0]), pi/2, 0] + grip,
    start + [math.atan2(start[1], start[0]), pi/2, 0] + grip,
    start + [math.atan2(start[1], start[0]), pi/2, 0] + [1],
]

full_radial_bounds = np.array([[0.1, -pi/2], [0.35, pi/2]])



# Various datasets for testing different aspects of the model. Each dataset is a dictionary containing the coordinates for the movement, the instruction for the movement, and the folder name for storing the data.
datasets = [
    # {"Coordinates": np.array([
    #     known_coordinates["center_mid_raised"] + gripper_state["release"],
    #     known_coordinates["center_mid_grounded"] + gripper_state["release"],
    #     known_coordinates["center_mid_grounded"] + gripper_state["hold"],
    #     known_coordinates["center_mid_raised"] + gripper_state["hold"],
    # ])},
    # {"Coordinates": np.array([
    #     modify_coords(known_coordinates["center_mid_raised"], 0.18, -0.07, 0.1) + gripper_state["release"],
    #     modify_coords(known_coordinates["center_mid_grounded"], 0.1, -0.07, 0.02) + gripper_state["release"],
    #     modify_coords(known_coordinates["center_mid_grounded"], 0.1, -0.07, 0.02) + gripper_state["hold"],
    #     modify_coords(known_coordinates["center_mid_raised"], 0.18, -0.07, 0.05) + gripper_state["hold"],
    #     modify_coords(known_coordinates["center_mid_raised"], 0.18, 0.07, 0.05) + gripper_state["hold"],
    #     modify_coords(known_coordinates["center_mid_raised"], 0.1, 0.07, 0.05) + gripper_state["release"]
    # ])},
    # {"Coordinates": np.array([
    #     known_coordinates["center_mid_raised"] + gripper_state["release"],
    #     known_coordinates["center_mid_grounded"] + gripper_state["release"],
    #     known_coordinates["center_mid_grounded"] + gripper_state["hold"],
    #     known_coordinates["center_mid_raised"] + gripper_state["hold"],
    #     known_coordinates["center_close_raised"] + gripper_state["hold"],
    #     known_coordinates["center_close_grounded"] + gripper_state["hold"],
    #     known_coordinates["center_close_grounded"] + gripper_state["release"],
    # ]),
    # "Instruction": "pick"
    # },
    # {"Coordinates": np.array([
    #     known_coordinates["center_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     known_coordinates["center_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     known_coordinates["center_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["center_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["center_close_raised"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["center_close_grounded"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["center_close_grounded"] + known_orientations["vertical_forward"] + gripper_state["release"],
    # ]),
    # "Instruction": "pick up the yellow block and put it closer to the robot",
    # "Folder": "block_center_mid_to_center_close_ios"
    # },
    # {"Coordinates": np.array([
    #     known_coordinates["center_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     known_coordinates["center_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     known_coordinates["center_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["center_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    # ]),
    # "Instruction": "pick up the yellow block",
    # "Folder": "pick_up_center_block_ios"
    # },
    # {"Coordinates": np.array([
    #     known_coordinates["left_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     known_coordinates["left_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     known_coordinates["left_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["left_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["right_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["right_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     known_coordinates["right_mid_grounded"] + known_orientations["vertical_forward"] + gripper_state["release"],
    # ]),
    # "Instruction": "pick up the yellow block and move it to the right",
    # "Folder": "block_left_mid_to_right_mid_ios"
    # },
    # {"Coordinates": np.array([
    #     [0.12, 0.12, 0.02] + [0, pi, 0] + [0],
    #     [0.20, 0.12, 0.02] + [0, pi, 0] + [0],
    #     [0.12, 0.12, 0.02] + [0, pi, 0] + [0],
    #     [0.12, 0.22, 0.02] + [0, pi, 0] + [0],
    #     [0.12, 0.03, 0.02] + [0, pi, 0] + [0],
    #     [0.12, 0.12, 0.02] + [0, pi, 0] + [0],
    #     [0.12, 0.12, 0.13] + [0, pi, 0] + [0],
    #     [0.12, 0.12, 0.13] + [0, pi, 145*deg2rad] + [0],
    #     [0.12, 0.12, 0.13] + [0, pi, -113*deg2rad] + [0],
    # ]),
    # "Instruction": "pick up the yellow block and move it to the right",
    # "Folder": "presentation_example"
    # },
    # {"Coordinates": np.array([
    #     known_coordinates["left_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     modify_coords(known_coordinates["left_mid_grounded"], 0.14, 0.12, 0.08) + known_orientations["side_forward"] + gripper_state["release"],
    #     modify_coords(known_coordinates["left_mid_grounded"], 0.14, 0.12, 0.08) + known_orientations["side_forward"] + gripper_state["1cm"],
    #     modify_coords(known_coordinates["left_mid_raised"], 0.14, 0.12, 0.13) + known_orientations["side_forward"] + gripper_state["1cm"],
    #     modify_coords(known_coordinates["right_mid_raised"], 0.14, -0.12, 0.13) + known_orientations["side_forward"] + gripper_state["1cm"],
    #     modify_coords(known_coordinates["right_mid_grounded"], 0.14, -0.12, 0.08) + known_orientations["side_forward"] + gripper_state["1cm"],
    #     modify_coords(known_coordinates["right_mid_grounded"], 0.14, -0.12, 0.08) + known_orientations["side_forward"] + gripper_state["release"],
    # ]),
    # "Instruction": "pick up the flask and move it to the right",
    # "Folder": "flask_left_mid_to_right_mid_side_grab_ios"
    # },
    # {"Coordinates": np.array([
    #     known_coordinates["right_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     modify_coords(known_coordinates["right_mid_grounded"], 0.14, -0.12, 0.01) + known_orientations["vertical_forward"] + gripper_state["release"],
    #     modify_coords(known_coordinates["right_mid_grounded"], 0.14, -0.12, 0.01) + known_orientations["vertical_forward"] + gripper_state["1cm"],
    #     modify_coords(known_coordinates["right_mid_raised"], 0.14, -0.12, 0.1) + known_orientations["vertical_forward"] + gripper_state["1cm"],
    #     modify_coords(known_coordinates["left_mid_raised"], 0.14, 0.12, 0.1) + known_orientations["vertical_forward"] + gripper_state["1cm"],
    #     modify_coords(known_coordinates["left_mid_grounded"], 0.14, 0.12, 0.01) + known_orientations["vertical_forward"] + gripper_state["1cm"],
    #     modify_coords(known_coordinates["left_mid_grounded"], 0.14, 0.12, 0.01) + known_orientations["vertical_forward"] + gripper_state["release"],
    # ]),
    # "Instruction": "pick up the screwdriver and move it to the left",
    # "Folder": "screwdriver_right_mid_to_left_mid_ios"
    # },
    # {"Coordinates": np.array([
    #     known_coordinates["center_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     [0.14, 0, 0.01] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     [0.14, 0, 0.01] + known_orientations["vertical_forward"] + gripper_state["1cm"],
    #     [0.14, 0, 0.1] + known_orientations["vertical_forward"] + gripper_state["1cm"],
    #     modify_coords(known_coordinates["left_mid_raised"], 0.14, 0.12, 0.1) + known_orientations["vertical_forward"] + gripper_state["1cm"],
    #     modify_coords(known_coordinates["left_mid_grounded"], 0.14, 0.12, 0.01) + known_orientations["vertical_forward"] + gripper_state["1cm"],
    #     modify_coords(known_coordinates["left_mid_grounded"], 0.14, 0.12, 0.01) + known_orientations["vertical_forward"] + gripper_state["release"],
    # ]),
    # "Instruction": "pick up the battery and move it to the left",
    # "Folder": "battery_center_mid_to_left_mid_ios"
    # },
    # {"Coordinates": np.array([
    #     known_coordinates["center_mid_raised"] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     [0.14, 0, 0.01] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     [0.14, 0, 0.01] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     [0.14, 0, 0.1] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     modify_coords(known_coordinates["left_mid_raised"], 0.14, -0.12, 0.1) + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     modify_coords(known_coordinates["left_mid_grounded"], 0.14, -0.12, 0.01) + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     modify_coords(known_coordinates["left_mid_grounded"], 0.14, -0.12, 0.01) + known_orientations["vertical_forward"] + gripper_state["release"],
    # ]),
    # "Instruction": "pick up the rock and move it to the right",
    # "Folder": "rock_center_mid_to_right_mid_ios"
    # },
    # {"Coordinates": np.array(set_coords([0.14, 0, 0.01], [0.14, -0.12, 0.01], known_orientations["vertical_forward"], known_orientations["vertical_forward"], gripper_state["3cm"])),
    # "Instruction": "pick up the block and move it to the right",
    # "Folder": "block_center_mid_to_right_mid_ios"
    # },
    # {"Coordinates": np.array(set_coords([0.14, 0, 0.01], [0.14, 0.10, 0.01], known_orientations["vertical_forward"], known_orientations["vertical_forward"], gripper_state["3cm"])),
    # "Instruction": "pick up the block and move it to the left",
    # "Folder": "block_center_mid_to_left_mid_ios"
    # },
    # {"Coordinates": np.array(set_coords([0.1, 0, 0.005], [0.3, 0.0, 0.005], known_orientations["vertical_forward"], known_orientations["vertical_forward"], gripper_state["3cm"])),
    # "Instruction": "pick up the block and move it away from the robot",
    # "Folder": "block_center_close_to_center_far_ios"
    # },
    # {"Coordinates": np.array(set_coords([0.1, 0, 0.01], [0.3, 0.0, 0.02], known_orientations["vertical_forward"], gripper_state["3cm"])),
    # "Instruction": "pick up the rock and move it towards the robot",
    # "Folder": "rock_center_far_to_center_close_ios"
    # },
    # {"Coordinates": np.array(set_coords([0.08, 0.08, 0.02], [0.3, 0.3, 0.00], known_orientations["vertical_slight_left"], known_orientations["vertical_slight_left"], gripper_state["3cm"])),
    # "Instruction": "pick up the block and move it away from the robot",
    # "Folder": "block_left_close_to_left_far_ios"
    # },
    # {"Coordinates": np.array(set_coords([0.08, -0.08, 0.01], [0.27, -0.27, 0.00], known_orientations["vertical_slight_right"], gripper_state["1cm"])),
    # "Instruction": "pick up the battery and move it away from the robot",
    # "Folder": "battery_right_close_to_right_far_ios"
    # },
    # {"Coordinates": np.array(set_coords([0.14, 0, 0.01], [0.14, 0.14, 0.01], known_orientations["vertical_forward"], gripper_state["1cm"])),
    # "Instruction": "pick up the pencil and move it to the left",
    # "Folder": "pencil_center_mid_to_left_mid_ios"
    # },
    # {"Coordinates": np.array(set_coords([0.16, 0, 0.01], [0.16, -0.16, 0.01], known_orientations["vertical_forward"], gripper_state["1cm"])),
    # "Instruction": "pick up the screwdriver and move it to the right",
    # "Folder": "screwdriver_center_mid_to_right_mid_ios"
    # },
    # {"Coordinates": np.array(set_coords([0.1, 0, 0.01], [0.35, 0, 0.04], known_orientations["vertical_forward"], known_orientations["side_forward"], gripper_state["1cm"])),
    # "Instruction": "pick up the ball and move it away from the robot",
    # "Folder": "ball_center_close_to_center_far_ios"
    # },
    # {"Coordinates": np.array(set_coords([0.15, 0.14, 0.015], [0.15, -0.14, 0.015], known_orientations["vertical_forward"], known_orientations["vertical_forward"], gripper_state["1cm"])),
    # "Instruction": "pick up the screwdriver and move it to the right",
    # "Folder": "screwdriver_left_mid_to_right_mid_ios"
    # },
    # {"Coordinates": np.array(set_coords([0.15, 0.16, 0.09], [0.15, 0.0, 0.09], known_orientations["vertical_forward"], known_orientations["vertical_forward"], gripper_state["1cm"])),
    # "Instruction": "pick up the flask and move it to the center",
    # "Folder": "flasks_left_mid_to_center_mid_ios"
    # },
    # {"Coordinates": np.array(set_coords([0.11, -0.2, 0.015], [0.15, 0.0, 0.015], known_orientations["vertical_forward"], known_orientations["vertical_forward"], gripper_state["1cm"])),
    # "Instruction": "pick up the screwdriver and move it to the center",
    # "Folder": "screwdriver_right_mid_to_center_mid_ios"
    # },
    # {"Coordinates": np.array(set_coords([0.1, -0.11, 0.04], [0.3, -0.3, 0.04], known_orientations["vertical_slight_right"], known_orientations["vertical_slight_right"], gripper_state["1cm"])),
    # "Instruction": "pick up the battery and move it away from the robot",
    # "Folder": "battery_upright_right_close_to_right_far_ios"
    # }
    # {"Coordinates": np.array(set_coords([0.1, -0.1, 0.01], [0.3, -0.1, 0.01], known_orientations["vertical_forward"], known_orientations["vertical_forward"], gripper_state["3cm"])),
    # "Instruction": "pick up the block and move it away from the robot",
    # "Folder": "block_close_right_to_far_right_ios"
    # }
    # {"Coordinates": np.array(set_coords([0.09, -0.035, 0.01], [0.26, -0.035, 0.02], known_orientations["vertical_forward"], known_orientations["vertical_forward"], gripper_state["3cm"])),
    # "Instruction": "pick up the block closest to the robot and move it to the other block",
    # "Folder": "block_mid_center_to_far_center_ios_4"
    # }

    # {"Coordinates": np.array(set_coords([0.10, 0.065, 0.01], [0.24, -0.0, 0.02], known_orientations["vertical_forward"], known_orientations["vertical_forward"], gripper_state["3cm"])),
    # "Instruction": "pick up the block and move it towards the rock",
    # "Folder": "block_move_cluttered_ios_2"
    # }

    #  --------- DROP INTO CUP -------- 
    # {"Coordinates": np.array(set_coords_cup([0.14, 0.12, 0.01], [0.14, -0.12, 0.18], known_orientations["vertical_forward"], known_orientations["vertical_forward"], gripper_state["1cm"])),
    # "Instruction": "pick up the ball and put it in the cup",
    # "Folder": "ball_left_mid_drop_cup_right_mid_ios"
    # },
    # {"Coordinates": np.array(set_coords_cup([0.14, -0.12, 0.01], [0.14, 0.12, 0.15], known_orientations["vertical_forward"], known_orientations["vertical_forward"], gripper_state["1cm"])),
    # "Instruction": "pick up the battery and put it in the cup",
    # "Folder": "battery_right_mid_drop_cup_left_mid_ios"
    # },
    # {"Coordinates": np.array(set_coords_cup([0.16, 0.0, 0.01], [0.0001, 0.22, 0.15], known_orientations["vertical_forward"], known_orientations["vertical_left"], gripper_state["3cm"])),
    # "Instruction": "pick up the rock and put it in the cup",
    # "Folder": "rock_center_mid_drop_cup_extreme_left_mid_ios"
    # },
    # {"Coordinates": np.array(set_coords_cup([0.16, 0.0, 0.01], [0.0001, -0.22, 0.15], [0, 180, 0], [0, 180, -90], gripper_state["1cm"])),
    # "Instruction": "pick up the ball and put it in the cup",
    # "Folder": "ball_center_mid_drop_cup_extreme_right_mid_ios"
    # },
    # {"Coordinates": np.array(set_coords_cup([0.0001, 0.22, -0.01], [0.25, -0.2, 0.18], [0, 180, 90], [0, 90, 0], gripper_state["3cm"])),
    # "Instruction": "pick up the rock and put it in the cup",
    # "Folder": "rock_extreme_left_mid_drop_cup_right_far_ios"
    # },
    # {"Coordinates": np.array(set_coords_cup([0.0001, -0.22, -0.01], [0.25, 0.2, 0.2], known_orientations["vertical_right"], known_orientations["side_forward"], gripper_state["1cm"])),
    # "Instruction": "pick up the ball and put it in the cup",
    # "Folder": "ball_extreme_right_mid_drop_cup_left_far_ios"
    # },
    # {"Coordinates": np.array(set_coords_cup([0.18, -0.26, 0.025], [0.18, -0.13, 0.2], known_orientations["vertical_right"], known_orientations["vertical_slight_right"], gripper_state["1cm"])),
    # "Instruction": "pick up the ball and put it in the cup",
    # "Folder": "ball_right_far_drop_cup_right_mid_ios"
    # },
    # {"Coordinates": np.array([
    #     [0.08, -0.26, 0.1] + known_orientations["vertical_right"] + [0.6],
    #     [0.15, -0.26, 0.1] + known_orientations["vertical_right"] + [0.6],
    #     [0.15, -0.26, 0.025] + known_orientations["vertical_right"] + [0.6],
    #     [0.15, -0.26, 0.025] + known_orientations["vertical_right"] + gripper_state["1cm"],
    #     [0.15, -0.26, 0.2] + known_orientations["vertical_right"] + gripper_state["1cm"],
    #     [0.18, -0.13, 0.2] + known_orientations["vertical_right"] + gripper_state["1cm"],
    #     [0.18, -0.13, 0.2] + known_orientations["vertical_right"] + gripper_state["release"],
    #     ]),
    # "Instruction": "pick up the ball and put it in the cup",
    # "Folder": "ball_right_far_drop_cup_right_mid_ios"
    # },
    # {"Coordinates": np.array(set_coords_cup([0.25, -0.26, 0.025], [0.3, 0, 0.25], known_orientations["vertical_slight_right"], known_orientations["vertical_forward"], gripper_state["1cm"])),
    # "Instruction": "pick up the ball and put it in the cup",
    # "Folder": "ball_right_far_drop_cup_center_far_ios"
    # },
    # {"Coordinates": np.array(set_coords_cup([0.09, 0, 0.025], [0.18, -0.05, 0.15], known_orientations["vertical_forward"], known_orientations["vertical_forward"], gripper_state["1cm"])),
    # "Instruction": "pick up the ball and put it in the cup",
    # "Folder": "ball_center_close_drop_cup_right_mid_ios"
    # },
    # {"Coordinates": np.array(set_coords_cup([0.17, 0, 0.02], [0.25, 0.14, 0.22], known_orientations["vertical_forward"], known_orientations["side_forward"], gripper_state["1cm"])),
    # "Instruction": "pick up the ball and put it in the cup",
    # "Folder": "ball_center_mid_drop_cup_left_far_ios"
    # },
    # {"Coordinates": np.array(set_coords_cup([0.2, -0.15, 0.02], [0.12, 0.1, 0.22], known_orientations["vertical_slight_right"], known_orientations["vertical_slight_left"], gripper_state["1cm"])),
    # "Instruction": "pick up the ball and put it in the cup",
    # "Folder": "ball_right_mid_drop_cup_left_mid_ios"
    # },
    # {"Coordinates": np.array(set_coords_cup([0.08, -0.08, 0.02], [0.01, -0.25, 0.20], known_orientations["vertical_slight_right"], known_orientations["vertical_right"], gripper_state["1cm"])),
    # "Instruction": "pick up the ball and put it in the cup",
    # "Folder": "ball_right_close_drop_cup_extreme_right_mid_ios"
    # },
    # {"Coordinates": np.array([
    #     [0.14, 0.14, 0.12] + known_orientations["vertical_left"] + gripper_state["release"],
    #     [0.14, 0.14, 0.02] + known_orientations["vertical_left"] + gripper_state["release"],
    #     [0.14, 0.035, 0.015] + known_orientations["vertical_left"] + gripper_state["release"],
    #     [0.14, 0.035, 0.015] + known_orientations["vertical_left"] + gripper_state["1cm"],
    #     [0.14, 0.05, 0.17] + known_orientations["vertical_left"] + gripper_state["1cm"],
    #     [0.14, -0.03, 0.17] + known_orientations["vertical_forward"] + gripper_state["1cm"],
    #     [0.14, -0.03, 0.17] + known_orientations["vertical_forward"] + gripper_state["release"],
    # ]),
    # "Instruction": "pick up the ball and put it in the cup",
    # "Folder": "ball_center_mid_drop_cup_center_mid_ios"
    # },
    # {"Coordinates": np.array(set_coords_cup([0.08, 0.2, 0.02], [0.14, -0.08, 0.15], known_orientations["vertical_slight_left"], known_orientations["vertical_forward"], gripper_state["1cm"])),
    # "Instruction": "pick up the ball and put it in the cup",
    # "Folder": "ball_left_mid_drop_cup_left_mid_ios"
    # },
    # {"Coordinates": np.array(set_coords_cup([0.24, 0.08, 0.02], [0.26, -0.09, 0.25], known_orientations["vertical_slight_left"], known_orientations["vertical_forward"], gripper_state["1cm"])),
    # "Instruction": "pick up the ball and put it in the cup",
    # "Folder": "ball_left_mid_drop_cup_right_mid_ios"
    # },
    # {"Coordinates": np.array(set_coords_cup([0.08, 0.17, 0.02], [0.08, -0.17, 0.2], known_orientations["vertical_slight_left"], known_orientations["vertical_slight_right"], gripper_state["1cm"])),
    # "Instruction": "pick up the ball and put it in the cup",
    # "Folder": "ball__extreme_left_mid_drop_cup_extreme_right_mid_ios"
    # },
    # {"Coordinates": np.array(set_coords_cup([0.24, 0.0, 0.015], [0.12, 0, 0.15], known_orientations["vertical_forward"], known_orientations["vertical_forward"], gripper_state["1cm"])),
    # "Instruction": "pick up the ball and put it in the cup",
    # "Folder": "ball_center_mid_drop_cup_center_close_cluttered_ios"
    # },
    # {"Coordinates": np.array(set_coords_cup([0.28, 0.16, 0.03], [0.01, 0.14, 0.20], known_orientations["vertical_slight_left"], known_orientations["side_left"], gripper_state["1cm"])),
    # "Instruction": "pick up the ball and put it in the cup",
    # "Folder": "pencil_left_far_drop_cup_extreme_left_close_cluttered_ios"
    # },
    # {"Coordinates": np.array(set_coords_cup([0.32, 0.24, 0.04], [0.01, 0.14, 0.17], known_orientations["vertical_slight_left"], known_orientations["side_left"], gripper_state["3cm"])),
    # "Instruction": "pick up the left block and put it in the cup",
    # "Folder": "block_left_far_drop_cup_extreme_left_close_cluttered_ios"
    # },
    # {"Coordinates": np.array(set_coords_cup([0.15, -0.11, 0.035], [0.01, 0.14, 0.15], known_orientations["vertical_slight_right"], known_orientations["side_left"], gripper_state["1cm"])),
    # "Instruction": "pick up the battery and put it in the cup",
    # "Folder": "battery_right_mid_drop_cup_extreme_left_close_cluttered_ios"
    # },
    # {"Coordinates": np.array(set_coords_cup([0.26, -0.21, 0.04], [0.24, 0.27, 0.25], known_orientations["vertical_slight_right"], known_orientations["side_slight_left"], gripper_state["3cm"])),
    # "Instruction": "pick up the rock and put it in the cup",
    # "Folder": "rock_right_far_drop_cup_left_far_cluttered_ios"
    # },
    # {"Coordinates": np.array(set_coords_cup([0.22, -0.14, 0.02], [0., -0.13, 0.2], known_orientations["vertical_slight_left"], known_orientations["side_right"], gripper_state["1cm"])),
    # "Instruction": "pick up the screwdriver and put it in the cup",
    # "Folder": "screwdriver_right_mid_drop_cup_left_close_cluttered_ios"
    # },
    # {"Coordinates": np.array(set_coords_cup([0.22, -0.14, 0.02], [0., -0.13, 0.2], known_orientations["vertical_slight_left"], known_orientations["side_right"], gripper_state["1cm"])),
    # "Instruction": "pick up the screwdriver and put it in the cup",
    # "Folder": "screwdriver_right_mid_drop_cup_left_close_cluttered_ios"
    # },
    # {"Coordinates": np.array(set_coords_cup([0.27, 0.17, 0.02], [0., -0.13, 0.2], [-30*deg2rad, 180*deg2rad, 0], known_orientations["side_right"], gripper_state["1cm"])),
    # "Instruction": "pick up the screwdriver and put it in the cup",
    # "Folder": "pencil_left_mid_drop_cup_right_close_cluttered_ios"
    # },
    # {"Coordinates": np.array(set_coords_cup([0.1, 0.2, 0.02], [0., -0.13, 0.2], known_orientations["vertical_forward"], known_orientations["side_right"], gripper_state["1cm"])),
    # "Instruction": "pick up the blue ball and put it in the cup",
    # "Folder": "ball_left_mid_drop_cup_right_close_cluttered_ios"
    # },
    # {"Coordinates": np.array(set_coords_cup([0.05, 0.26, 0.02], [0., -0.13, 0.2], known_orientations["vertical_slight_left"], known_orientations["side_right"], gripper_state["3cm"])),
    # "Instruction": "pick up the leftmost block and put it in the cup",
    # "Folder": "block_left_mid_drop_cup_right_close_cluttered_ios"
    # },
    # {"Coordinates": np.array(
    #     place_block([0.2, -0.2, 0.00], [0.001, -0.35, 0.05], [-pi/2, pi, 0], [-pi/2, pi, 0], [0.2]) +
    #     place_block([0.3, -0.10, 0.00], [0.001, -0.35, 0.05], [-pi/4, pi, 0], [-pi/2, pi, 0], [0.2])),
    #     # set_coords_cup([0.18, 0.20, 0.00], [0.001, -0.3, 0.25], [pi/4, pi, 0], [-pi/2, pi/2, 0], [0.2])),
    # "Instruction": "pick up the trash and put it in the bin",
    # "Folder": "trash_drop_cup_cluttered_1"
    # },
    # {"Coordinates": np.array(
    #     set_coords_cup([0.14, 0.24, 0.00], [0.18, -0.25, 0.23], [pi/4, pi, 0], [-pi/4, pi/2, 0.3], [0.2]) + 
    #     set_coords_cup([0.25, -0.03, 0.015], [0.18, -0.25, 0.25], [0, pi, 0], [-pi/4, pi/2, 0.3], [0.2]) + 
    #     set_coords_cup([0.07, -0.18, 0.01], [0.18, -0.25, 0.25], [-pi/4, pi, 0], [-pi/2, pi/2, 0.3], [0.2])),
    # "Instruction": "pick up the trash and put it in the cup",
    # "Folder": "trash_drop_cup_2"
    # },


    # {"Coordinates": np.array([
    #     [0.15, -0.15, 0.14] + [-pi/4, pi/2, 0] + [0.8],
    #     [0.19, -0.19, 0.14] + [-pi/4, pi/2, 0] + [0.8],
    #     [0.19, -0.19, 0.14] + [-pi/4, pi/2, 0] + [0.25],
    #     [0.19, -0.19, 0.2] + [-pi/4, pi/2, 0] + [0.25],
    #     [0.24, -0.0, 0.2] + [0, pi/2, 0] + [0.25],
    #     [0.24, -0.0, 0.2] + [-pi/2, -0.17, -pi/2] + [0.25],
    #     [0.24, -0.0, 0.2] + [-pi/2, -0.17, -pi/2] + [0.25],
    #     [0.24, -0.0, 0.2] + [0,pi/2,0] + [0.25],
    #     [0.19, -0.19, 0.2] + [-pi/4, pi/2, 0] + [0.25],
    #     [0.19, -0.19, 0.14] + [-pi/4, pi/2, 0] + [0.25],
    #     [0.19, -0.19, 0.14] + [-pi/4, pi/2, 0] + [0.8],
    # ]),
    # "Instruction": "pour the water bottle into the cup",
    # "Folder": "water_bottle_pour_cup"
    # }
    # {"Coordinates": np.array([
    #     [0.20, 0.1, 0.12] + [pi/4, pi/2, 0] + [0.8],
    #     [0.26, 0.16, 0.12] + [pi/4, pi/2, 0] + [0.8],
    #     [0.26, 0.16, 0.12] + [pi/4, pi/2, 0] + [0.25],
    #     [0.26, 0.16, 0.2] + [pi/4, pi/2, 0] + [0.25],
    #     [0.16, -0.0, 0.2] + [0, pi/2, 0] + [0.25],
    #     [0.16, -0.0, 0.2] + [pi*0.6, -0.17, pi*0.6] + [0.25],
    #     [0.16, -0.0, 0.2] + [pi*0.6, -0.17, pi*0.6] + [0.25],
    #     [0.16, -0.0, 0.2] + [0,pi/2,0] + [0.25],
    #     [0.26, 0.16, 0.2] + [pi/4, pi/2, 0] + [0.25],
    #     [0.26, 0.16, 0.12] + [pi/4, pi/2, 0] + [0.25],
    #     [0.26, 0.16, 0.12] + [pi/4, pi/2, 0] + [0.8],
    # ]),
    # "Instruction": "pour the water bottle into the cup",
    # "Folder": "water_bottle_pour_cup_2"
    # }
    {"Coordinates": np.array(pour_water([0.05,0.3,0.14], [0.22,-0.23,0.25], [-pi/4,pi/2,0],[pi*0.25, -0.3, pi*0.5], [0.2])),
        "Instruction": "pour the water bottle into the cup",
        "Folder": "water_bottle_pour_cup_26"
    },
    
    # {"Coordinates": np.array([
    #     [0.24, -0.1, 0.07] + [-pi/2, 0,0] + [0.5],
    #     [0.24, -0.2, 0.07] + [-pi/2, 0,0] + [0.5],
    #     [0.24, -0.2, 0.07] + [-pi/2, 0,0] + [0.2],
    #     [0.24, 0, 0.2] + [-pi/2, 0,0] + [0.2],
    # ]),
    # "Instruction": "pull the tin foil out",
    # "Folder": "tin_foil_pull"
    # },
    #  ----- STACKING ------- 
    # {"Coordinates": np.array(set_coords([0.1, 0.1, 0.01], [0.14, 0.0, 0.04], known_orientations["vertical_forward"], gripper_state["1cm"])),
    # "Instruction": "stack the ball on the block",
    # "Folder": "ball_left_mid_stack_block_center_mid_ios"
    # },
    # {"Coordinates": np.array([
    #     [0.15, 0.1, 0.1] + known_orientations["vertical_forward"] + [1],
    #     [0.15, 0.1, 0.01] + known_orientations["vertical_forward"] + [1],
    #     [0.15, 0.1, 0.01] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     [0.15, 0.1, 0.01] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     [0.3, 0.0, 0.2] + known_orientations["side_forward"] + gripper_state["3cm"],
    #     [0.3, 0.0, 0.1] + known_orientations["side_forward"] + gripper_state["3cm"],
    #     [0.3, 0.0, 0.1] + known_orientations["side_forward"] + gripper_state["release"],
    #     [0.3, 0.0, 0.2] + known_orientations["side_forward"] + gripper_state["release"],
    #     [0.15, -0.1, 0.2] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     [0.15, -0.1, 0.03] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     [0.15, -0.1, 0.03] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     [0.15, -0.1, 0.15] + known_orientations["vertical_forward"] + gripper_state["3cm"],
    #     [0.3, 0., 0.2] + known_orientations["side_forward"] + gripper_state["3cm"],
    #     [0.3, 0., 0.16] + known_orientations["side_forward"] + gripper_state["3cm"],
    #     [0.3, 0., 0.16] + known_orientations["side_forward"] + gripper_state["release"],
    #     [0.01, -0.17, 0.2] + known_orientations["vertical_right"] + gripper_state["release"],
    #     [0.01, -0.17, 0.03] + known_orientations["vertical_right"] + gripper_state["release"],
    #     [0.01, -0.17, 0.03] + known_orientations["vertical_right"] + gripper_state["3cm"],
    #     [0.01, -0.17, 0.24] + known_orientations["vertical_right"] + gripper_state["3cm"],
    #     [0.3, 0., 0.24] + known_orientations["side_forward"] + gripper_state["3cm"],
    #     [0.3, 0., 0.2] + known_orientations["side_forward"] + gripper_state["3cm"],
    #     [0.3, 0., 0.2] + known_orientations["side_forward"] + gripper_state["release"],
    # ]),
    # "Instruction": "stack the blocks on top of each other",
    # "Folder": "block_tower_stack"
    # },
    # {"Coordinates": np.array([
    #     [0.24, 0.0, 0.15] + [-pi/4, pi, 0] + [1],
    #     [0.24, 0.0, 0.03] + [-pi/4, pi, 0] + [1],
    #     [0.24, 0.0, 0.03] + [-pi/4, pi, 0] + [0.3],
    #     [0.24, 0.0, 0.03] + [0, pi, 0] + [0.3],
    #     [0.25, 0.0, 0.15] + [0, pi, 0] + [1],
    #     [0.17, -0.28, 0.15] + [-pi/4, 3*pi/4, 0] + [1],
    #     [0.17, -0.28, 0.04] + [-pi/4, 3*pi/4, 0] + [1],
    #     [0.17, -0.28, 0.04] + [-pi/4, 3*pi/4, 0] + [0.3],
    #     [0.17, -0.28, 0.15] + [-pi/4, 3*pi/4, 0] + [0.3],
    #     [0.24, 0.0, 0.2] + [0, 3*pi/4, 0] + [0.3],
    #     [0.24, 0.0, 0.075] + [0, 3*pi/4, 0] + [0.3],
    #     [0.24, 0.0, 0.075] + [0, 3*pi/4, 0] + [1],
    #     [0.24, 0.0, 0.2] + [0, 3*pi/4, 0] + [1],
    #     [0.135, 0.19, 0.2] + [pi/2, 3*pi/4, 0] + [1],
    #     [0.135, 0.19, 0.02] + [pi/2, 3*pi/4, 0] + [1],
    #     [0.135, 0.19, 0.02] + [pi/2, 3*pi/4, 0] + [0.3],
    #     [0.135, 0.19, 0.2] + [pi/2, pi, 0] + [0.3],
    #     [0.24, 0.0, 0.2] + [0, 3*pi/4, 0] + [0.3],
    #     [0.24, 0.0, 0.115] + [0, 3*pi/4, 0] + [0.3],
    #     [0.24, 0.0, 0.115] + [0, 3*pi/4, 0] + [1],
    # ]),
    # "Instruction": "stack the blocks on top of the middle block",
    # "Folder": "block_tower_stack_12"
    # },
    # (10,13) (21.5, 12) (30, 10)
    # {"Coordinates": np.array(
    #     place_block([0.215, -0.12, 0.03], [0.10, -0.13, 0.075], [-pi/4, 3*pi/4, 0], [-pi/4, 3*pi/4, 0], [0.3]) + 
    #     place_block([0.3, 0.10, 0.05], [0.10, -0.13, 0.115], [pi/4, 3*pi/4, 0], [-pi/4, 3*pi/4, 0], [0.3]) 
    # ),
    # "Instruction": "stack the blocks on top of the middle block",
    # "Folder": "block_tower_stack_15"
    # },
    # {"Coordinates": np.array(
    #     place_block([0.001, -0.2, 0.025], [0.25, 0, 0.075], [-pi/2, 3*pi/4, 0], [0, 3*pi/4, 0], [0.3]) + 
    #     place_block([0.001, -0.25, 0.03], [0.25, 0, 0.115], [-pi/2, 3*pi/4, 0], [0, 3*pi/4, 0], [0.3]) + 
    #     place_block([0.13, -0.13, 0.025], [0.25, 0, 0.15], [-pi/4, 3*pi/4, 0], [0, 3*pi/4, 0], [0.3]) +
    #     place_block([0.13, 0.13, 0.025], [0.25, 0, 0.19], [pi/4, 3*pi/4, 0], [0, 3*pi/4, 0], [0.3]) +
    #     place_block([0.001, 0.2, 0.015], [0.25, 0, 0.235], [pi/2, pi, 0], [0, pi/2, 0], [0.3]) 
    # ),
    # "Instruction": "stack the blocks on top of the middle block",
    # "Folder": "6_block_tower_stack_2"
    # },
    # {"Coordinates": np.array(
    #     place_block([0.001, -0.2, 0.005], [0.16, 0, 0.02], [-pi/2, pi, 0], [0, 150*deg2rad, 0], [0.2]) +
    #     [[0.15, 0, 0.15, 0, 150*deg2rad, 0, 0.25]] +
    #     [[0.15, 0, 0.00, 0, 150*deg2rad, 0, 0.25]] +
    #     [[0.15, 0, 0.15, 0, 150*deg2rad, 0, 0.25]]
    # ),
    # "Instruction": "put the battery into the battery charger",
    # "Folder": "battery_charge_0"
    # },

    #   ------ ROTATION --------
    # {"Coordinates": np.array(set_coords_rotation([0.2, 0., 0.02], [0.2, 0.0, 0.02], known_orientations["vertical_left"], known_orientations["vertical_right"], gripper_state["3cm"])),
    # "Instruction": "turn the block around",
    # "Folder": "block_center_mid_rotate_180_ios"
    # },
    # {"Coordinates": np.array(set_coords_rotation([0.26, 0., 0.02], [0.26, 0.0, 0.02], known_orientations["side_forward"], [90*pi, -90*pi, 0], gripper_state["3cm"])),
    # "Instruction": "flip the block upside down",
    # "Folder": "block_center_mid_flip_upside_down_ios"
    # },
    # {"Coordinates": np.array(set_coords_rotation([0.26, 0., 0.02], [0.26, 0.0, 0.02], known_orientations["side_forward"], [0, 90*pi, 0], gripper_state["1cm"])),
    # "Instruction": "place the battery on its end",
    # "Folder": "battery_center_mid_upright_center_mid_ios"
    # },
    # ---------- Miscellaneous
    # {"Coordinates": np.array([
    #     [0.14, 0.16, 0.1] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     [0.14, 0.16, 0.02] + known_orientations["vertical_forward"] + gripper_state["release"],
    #     [0.14, 0.16, 0.02] + known_orientations["vertical_forward"] + gripper_state["1cm"],
    #     [0.14, 0.0, 0.05] + known_orientations["vertical_forward"] + gripper_state["1cm"],
    #     [0.12, 0.0, 0.05] + known_orientations["vertical_forward"] + gripper_state["1cm"],
    #     [0.26, 0.0, 0.05] + known_orientations["vertical_forward"] + gripper_state["1cm"],
    #     [0.14, 0.0, 0.05] + known_orientations["vertical_forward"] + gripper_state["1cm"],
    # ]),
    # "Instruction": "stab the creeper with the knife",
    # "Folder": "knife_stab"
    # },
    # {"Coordinates": np.array(set_coords([0.28, 0.0, 0.1], [0.28, 0.05, 0.1], known_orientations["side_forward"], known_orientations["side_forward"], [0.2])),
    # "Instruction": "pick up the water bottle and move it to the center",
    # "Folder": "weight_test"
    # },
    # {"Coordinates": np.array([
    #     [0.18, 0.14, 0.18, pi/6, pi, 0, 0.3],
    #     [0.18, 0.14, 0.08, pi/6, pi, 0, 0.3],
    #     [0.25, 0.0, 0.08, pi/6, pi, 0, 0.3],
    # ]),
    # "Instruction": "open the drawer",
    # "Folder": "drawer_open_3"
    # },
    # {"Coordinates": np.array([
    #     [0.26, 0.0, 0.18, pi/6, pi, 0, 0.3],
    #     [0.26, 0.0, 0.08, pi/6, pi, 0, 0.3],
    #     [0.14, 0.16, 0.08, pi/6, pi, 0, 0.3],
    # ]),
    # "Instruction": "close the drawer",
    # "Folder": "drawer_close_2"
    # },
    # {"Coordinates": np.array([
    #     [0.18, 0.14, 0.18, pi/6, pi, 0, 0.3],
    #     [0.18, 0.14, 0.08, pi/6, pi, 0, 0.3],
    #     [0.25, 0.0, 0.08, pi/6, pi, 0, 0.3],
    #     [0.25, 0.0, 0.18, pi/6, pi, 0, 0.3]] +
    #     set_coords_cup([.17,-.12,0.02], [0.2,0.14,0.2],[-pi/4,pi,0],[pi/4,pi,0], [0.2]) +
    #     [[0.26, 0.0, 0.18, pi/6, pi, 0, 0.3],
    #     [0.26, 0.0, 0.08, pi/6, pi, 0, 0.3],
    #     [0.14, 0.16, 0.08, pi/6, pi, 0, 0.3],
    # ]),
    # "Instruction": "put the trash into the drawer",
    # "Folder": "trash_in_drawer_5",
    # }
]

if __name__ == "__main__":
    pass