import random
import numpy as np
from dataset_collection_constants import *
from Bezier import Bezier


def get_random_float(low, high):
    return random.random() * (high - low) + low

def create_random_dataset(n:int):
    """m
    Creates a dataset of random movements for a given object. The movements are generated based on the last position of the object, ensuring that the new position is within specified bounds and not too close to the previous position. The dataset includes coordinates for the movement, an instruction for the movement, and a folder name for storing the data.
    Parameters:
    - n: Number of random movements to generate.
    Returns:
    - A list of dictionaries, each containing the coordinates for the movement, the instruction for the movement, and the folder name for storing the data.
    """
    rand_samples = n
    bounds = np.array([[0.1, -0.1],[0.29, 0.1]])
    min_dist = 0.04
    rand_ranges = np.random.random(rand_samples)
    datasets = []
    height = 0.005
    last_position = [0.11, 0.0, height]
    object_name = "block"

    for i in range(rand_samples):
        
        # choose axis of movement
        if random.random() < 0.75:   # deals with x axis (forward/backward)
            if last_position[0] - min_dist < bounds[0,0]:       # takes care if the current position is close to bound
                direction = "f"
            elif last_position[0] + min_dist > bounds[1,0]:
                direction = "b"
            elif random.random() < 0.25:
                direction = "f"
            else:
                direction = "b"
        else:
            if last_position[1] - min_dist < bounds[0,1]:       # takes care if the current position is close to bound
                direction = "l"
            elif last_position[1] + min_dist > bounds[1,1]:
                direction = "r"
            elif random.random() < 0.25:
                direction = "l"
            else:
                direction = "r"
        next_position = [0.15, 0, 0.02]
        instruction = "error"
        
        match direction:
            case "f":
                next_position = [get_random_float(last_position[0] + min_dist, bounds[1, 0]), last_position[1], height]
                instruction = "away from the robot"
            case "b":
                next_position = [get_random_float(bounds[0,0], last_position[0] - min_dist), last_position[1], height]
                instruction = "towards the robot"
            case "l":
                next_position = [last_position[0], get_random_float(last_position[1] + min_dist, bounds[1, 1]), height]
                instruction = "to the left"
            case "r":
                next_position = [last_position[0], get_random_float(bounds[0,1], last_position[1] - min_dist), height]
                instruction = "to the right"
        next_position[2] = 0.2 * pow(next_position[0] ** 2 + next_position[1] ** 2, 0.5) - 0.02 + height if pow(next_position[0] ** 2 + next_position[1] ** 2, 0.5) > 0.15 else height + 0.01
        datasets.append(
            {"Coordinates": np.array(set_coords(last_position, next_position, known_orientations["vertical_forward"], known_orientations["vertical_forward"], [0.25])),
            "Instruction": f"pick up the {object_name} and move it to a random location {instruction}",
            "Folder": f"{object_name}_center_random_{i+100}"
        })
        last_position = next_position
    return datasets

def check_bounds(position, inner_bound, outer_bound, radial_bound):
    mag = np.linalg.norm(position)
    if mag < radial_bound[0,0] or mag > radial_bound[1,0]:
        return False
    angle = math.atan2(position[1], position[0])
    if angle < -pi/2 or angle > pi/2:
        return False
    return True

def create_random_dataset_2(n:int=20, folder_start=0, object_name:str="block", height:float=0.01):
    """
    Creates a dataset of random movements for a given object. The movements are generated based on the last position of the object, ensuring that the new position is within specified bounds and not too close to the previous position. The dataset includes coordinates for the movement, an instruction for the movement, and a folder name for storing the data.
    Parameters:
    - n: Number of random movements to generate.
    - folder_start: Starting index for folder naming.
    - object_name: Name of the object being moved (used in instructions and folder names).
    - height: Height of the blocks.
    Returns:
    - A list of dictionaries, each containing the coordinates for the movement, the instruction for the movement, and the folder name for storing the data."""
    rand_samples = n
    last_orientation = [0, pi, 0]
    bounds = np.array([[0.001, -0.35],[0.35, 0.35]])
    inner_bound = np.array([[0.001, -0.1],[0.1, 0.1]])
    radial_bounds = np.array([[0.11, -pi/2],[0.3, pi/2]])
    min_dist = 0.08
    datasets = []
    # height = 0.01
    last_position = [0.115, 0.0, height]
    # object_name = "block"

    for i in range(rand_samples):
        choice = random.random()
        magnitude = np.linalg.norm(last_position[:2])
        angle = math.atan2(last_position[1], last_position[0])
        # choose axis of movement
        if choice < 0.4:   # deals with x axis (forward/backward)
            if not check_bounds(np.array([last_position[0] - min_dist, last_position[1]]), inner_bound, bounds, radial_bounds):       # takes care if the current position is close to bound
                direction = "f"
            elif not check_bounds(np.array([last_position[0] + min_dist, last_position[1]]), inner_bound, bounds, radial_bounds):
                direction = "b"
            elif random.random() < 0.5:
                direction = "f"
            else:
                direction = "b"
        elif choice < 0.8:
            if pow(last_position[0] ** 2 + last_position[1] ** 2, 0.5) - min_dist < radial_bounds[0,0]:
                direction = "a"
            elif pow(last_position[0] ** 2 + last_position[1] ** 2, 0.5) + min_dist > radial_bounds[1,0]:
                direction = "t"
            elif random.random() < 0.5:
                direction = "a"
            else:
                direction = "t"
        else:
            if not check_bounds(np.array([last_position[0], last_position[1] - min_dist]), inner_bound, bounds, radial_bounds):       # takes care if the current position is close to bound
                direction = "l"
            elif not check_bounds(np.array([last_position[0], last_position[1] + min_dist]), inner_bound, bounds, radial_bounds):
                direction = "r"
            elif random.random() < 0.5:
                direction = "l"
            else:
                direction = "r"
        next_position = [0.15, 0, 0.02]
        instruction = "error"
        magnitude = np.linalg.norm(last_position[:2])
        angle = math.atan2(last_position[1], last_position[0])
        match direction:
            case "f":
                next_position = [get_random_float(last_position[0] + min_dist, math.pow(radial_bounds[1,0] ** 2 - last_position[1] ** 2, 0.5)), last_position[1], height]
                instruction = "forwards"
            case "b":
                if last_position[1] < inner_bound[1,1] and last_position[1] > inner_bound[0,1]:
                    next_position = [get_random_float( math.pow(radial_bounds[0,0] ** 2 - last_position[1] ** 2, 0.5), last_position[0] - min_dist), last_position[1], height]
                else:
                    next_position = [get_random_float(bounds[0,0], last_position[0] - min_dist), last_position[1], height]
                instruction = "backwards"
            case "a":
                f = get_random_float(magnitude+min_dist, radial_bounds[1,0])
                next_position = [f*math.cos(angle), f*math.sin(angle), height]
                instruction = "away from the robot"
            case "t":
                f = get_random_float(radial_bounds[0,0], magnitude-min_dist)
                next_position = [f*math.cos(angle), f*math.sin(angle), height]
                instruction = "towards the robot"
            case "l":
                if last_position[0] < inner_bound[1,0] and last_position[1] < 0:
                    ch = random.random()
                    if ch < 0.3:
                        next_position = [last_position[0], get_random_float(last_position[1] + min_dist, inner_bound[0,1]), height]
                    else:
                        next_position = [last_position[0], get_random_float(inner_bound[1,1], math.pow(radial_bounds[1,0] ** 2 - last_position[0] ** 2, 0.5)), height]
                else:
                    next_position = [last_position[0], get_random_float(last_position[1] + min_dist, math.pow(radial_bounds[1,0] ** 2 - last_position[0] ** 2, 0.5)), height]
                instruction = "to the left"
            case "r":
                if last_position[0] < inner_bound[1,0] and last_position[1] > 0:
                    ch = random.random()
                    if ch < 0.3:
                        next_position = [last_position[0], get_random_float(inner_bound[1,1], last_position[1] - min_dist), height]
                    else:
                        print(math.pow(radial_bounds[1,0] ** 2 - last_position[0] ** 2, 0.5))
                        next_position = [last_position[0], get_random_float(-math.pow((radial_bounds[1,0] ** 2)  - ( last_position[0] ** 2), 0.5), inner_bound[0,1]), height]
                else:
                    next_position = [last_position[0], get_random_float(-math.pow((radial_bounds[1,0] ** 2)  - ( last_position[0] ** 2), 0.5), last_position[1] - min_dist), height]
                instruction = "to the right"
        if np.linalg.norm(next_position[:2]) > 0.21:
            next_position[2] += 0.01
        if np.linalg.norm(next_position[:2]) > 0.27:
            next_position[2] += 0.01
        # 0.3 -> 0.08
        # 0.25 -> 0.05
        datasets.append(
            {"Coordinates": np.array(set_coords(last_position, next_position, [angle, 3*pi/4, 0], [math.atan2(next_position[1], next_position[0]), 3*pi/4, 0], [0.2])),
            "Instruction": f"pick up the {object_name} and move it to a random location {instruction}",
            "Folder": f"{object_name}_random_{i+folder_start}"
        })  
        print(f"{direction}: {last_position} -> {next_position}")
        last_position = next_position
    return datasets

def generate_random_block_placements(n=3, min_dist=0.1, radial_bounds=np.array([[0.1, -pi/2],[0.3, pi/2]]), avoid_positions=None, height=0.02):
    """
    Generates random block placements within specified radial bounds while ensuring a minimum distance between blocks and optionally avoiding certain positions.
    Parameters:
    - n: Number of block placements to generate.
    - min_dist: Minimum distance required between any two blocks.
    - radial_bounds: Array specifying the radial bounds for block placement.
    - avoid_positions: Array of positions to avoid.
    - height: Height of the blocks.
    Returns:
    - List of block placements, where each placement is a list containing x, y, z coordinates and orientation.
    """
    positions = np.zeros((n,2))
    if avoid_positions:
        positions = np.vstack([positions, avoid_positions])
    # print(positions)
    results = []
    # height = 0.02
    for i in range(n):
        # random placement
        rand_magnitude = get_random_float(radial_bounds[0][0], radial_bounds[1][0])
        rand_angle = get_random_float(radial_bounds[0][1], radial_bounds[1][1])
        pos = np.array([rand_magnitude * math.cos(rand_angle), rand_magnitude * math.sin(rand_angle)])
        # print(pos, positions, np.sqrt(np.sum((positions - pos)**2,axis=1))  )
        while np.any(np.sqrt(np.sum((positions-pos)**2, axis=1)) < min_dist):
            # print(pos, positions, np.linalg.norm(pos - positions, axis=0))
            rand_magnitude = get_random_float(radial_bounds[0][0], radial_bounds[1][0])
            rand_angle = get_random_float(radial_bounds[0][1], radial_bounds[1][1])
            pos = np.array([rand_magnitude * math.cos(rand_angle),rand_magnitude * math.sin(rand_angle)])
            
        positions[i] = pos[:2]
        
        orientation = rand_angle + get_random_float(-pi/4, pi/4)
        results += [pos.tolist() + [height + height_adjust(pos)] + [orientation]]
        # print(results)

    return results


def create_random_block_stack_task(block_positions):
    """
    Creates a task for stacking blocks based on their positions. The task involves picking up each block and placing it on top of the previous one, with specific orientations based on their distance from the robot.
    Parameters:
    - block_positions: List of block positions, where each position is a list containing x, y, z coordinates and orientation.
    Returns:
    - List of coordinates for the stacking task, where each coordinate is a list containing x, y, z coordinates and orientation for the robot to follow.
    """
    grip = 0.3
    task = []
    for i in range(len(block_positions) - 1):
        start_orientation = [block_positions[i][3], pi if np.linalg.norm(block_positions[i][:2]) < 0.12 else 3*pi/4, 0]
        end_orientation = [block_positions[-1][3], pi/2 if np.linalg.norm(block_positions[i][:2]) < 0.12 else 3*pi/4, 0]
        task += place_block(block_positions[i][:3], block_positions[-1][:2] + [0.06 + 0.04*i + height_adjust(block_positions[-1][:2])], start_orientation[:2] + [start_orientation[2] + height_adjust(start_orientation[:2])], end_orientation, [grip])
    return task

def create_unstack_blocks_task(block_positions, stack_position):
    """
    Creates a task for unstacking blocks based on their positions. The task involves picking up each block from the stack and placing it in its original position, with specific orientations based on their distance from the robot.
    Parameters:
    - block_positions: List of block positions, where each position is a list containing x, y, z coordinates and orientation.
    - stack_position: Position of the stack, where the blocks are currently placed.
    Returns:
    - List of coordinates for the unstacking task, where each coordinate is a list containing x, y, z coordinates and orientation for the robot to follow.
    """
    task = []
    grip = 0.3
    for i in range(len(block_positions)):
        start_orientation = [stack_position[3], pi/2 if np.linalg.norm(block_positions[i][:2]) < 0.12 and i < len(block_positions) else 3*pi/4, 0]
        end_orientation = [block_positions[i][3], pi if np.linalg.norm(block_positions[i][:2]) < 0.12 and i < len(block_positions) else 3*pi/4, 0]
        task += place_block(stack_position[:2] + [0.025 + 0.045*(len(block_positions) - i - 1) + height_adjust(stack_position[:2])], block_positions[i][:3], start_orientation, end_orientation, [grip])
    return task

def create_random_block_stack_dataset(n=20, num_blocks=3, folder_start=0):
    """
    Creates a dataset of random block stacking and unstacking tasks. The dataset includes coordinates for both stacking and unstacking tasks, instructions for each task, and folder names for storing the data.
    Parameters:
    - n: Number of random block stacking and unstacking tasks to generate.
    - num_blocks: Number of blocks to be stacked/unstacked in each task.
    - folder_start: Starting index for folder naming.
    Returns:
    - A list of dictionaries, each containing the coordinates for the stacking and unstacking tasks, the instructions for each task, and the folder names for storing the data."""
    dataset = []
    block_stack = [0.25, 0, 0.02, 0]
    radial_bounds = np.array([[0.12, -pi/2],[0.3, pi/2]])
    for i in range(n):
        block_placements = generate_random_block_placements(num_blocks, 0.12, avoid_positions=block_stack[:2], radial_bounds=radial_bounds)
        dataset.append(
            {"Coordinates": np.array(create_unstack_blocks_task(block_placements, block_stack)),
            "Instruction": f"unstack the blocks",
            "Folder": f"block_unstack_random_{i+folder_start}"
        })
        dataset.append(
            {"Coordinates": np.array(create_random_block_stack_task(block_placements)),
            "Instruction": f"stack the blocks",
            "Folder": f"block_stack_random_{i+folder_start}"
        })
        block_stack = block_placements[-1]
    
    return dataset
# print(create_random_block_stack_dataset(3))


def create_random_bottle_pour_dataset(n=5, cup_position=[0.2, 0], radial_bounds=None, start_position=None, orientation1=[0,pi/2,0],orientation2=[-pi*0.6, -0.3, -pi*0.6]):
    """
    Creates a dataset of random bottle pouring tasks. The dataset includes coordinates for picking up the water bottle, moving it to a random location, and pouring it into a cup, along with instructions for each task and folder names for storing the data.
    Parameters:    
    - n: Number of random bottle pouring tasks to generate.
    - cup_position: Position of the cup where the water will be poured.
    - radial_bounds: Array specifying the radial bounds for random bottle placement.
    - start_position: Starting position of the water bottle. If None, a default position will be used.
    - orientation1: Orientation for picking up the water bottle.
    - orientation2: Orientation for pouring the water bottle into the cup.
    Returns:
    - A list of dictionaries, each containing the coordinates for the bottle pouring tasks, the instructions for each task, and the folder names for storing the data.
    """
    magnitude = get_random_float(0.12, 0.3)
    angle = get_random_float(-pi/2+0.001, -pi/8)
    if start_position is None:
        last_position = [0.25, 0.25, 0.12 + height_adjust([0.25,0.25])]
    else:
        last_position = start_position
    print(last_position)
    dataset = []
    if radial_bounds is None:
        radial_bounds = full_radial_bounds
    
    for i in range(n):
        magnitude = get_random_float(radial_bounds[0,0], radial_bounds[1,0])
        angle = get_random_float(radial_bounds[0,1], radial_bounds[1,1])
        next_position = generate_random_block_placements(1, 0.15, radial_bounds, cup_position, height=0.12)[0]
        next_position[2] += height_adjust(next_position[:2])
        
        # print(set_coords(last_position, next_position, [math.atan2(last_position[1], last_position[0]), pi/2, 0], [angle, pi/2, 0], [0.2]))
        # print(next_position)
        dataset.append({
            "Coordinates": np.array(set_coords(last_position, next_position[:3], [math.atan2(last_position[1], last_position[0]), pi/2, 0], [angle, pi/2, 0], [0.2])),
            "Instruction": "put the water bottle in a random location",
            "Folder": f"water_bottle_place_random_{i}"
        })
        # print(pour_water(last_position, next_position[:3],orientation1, orientation2, [0.2]))
        dataset.append({
            "Coordinates": np.array(pour_water(next_position[:3], cup_position + [0.25] ,orientation1, orientation2, [0.2])),
            "Instruction": "pour the water bottle into the cup",
            "Folder": f"water_bottle_pour_{i}"
        })
        last_position = next_position[:3]

    
    return dataset

def create_cup_drop_dataset(n=5, folder_start=0, cup_position=[0.2, 0], drop_off_position=[0.15,0], radial_bounds=None, object_name="ball"):
    """
    Creates a dataset of random cup dropping tasks. The dataset includes coordinates for picking up an object, moving it to a random location, and dropping it into a cup, along with instructions for each task and folder names for storing the data.
    Parameters:
    - n: Number of random cup dropping tasks to generate.
    - folder_start: Starting index for folder naming.
    - cup_position: Position of the cup where the object will be dropped.
    - drop_off_position: Position where the object will be picked up from.
    - radial_bounds: Array specifying the radial bounds for random object placement.
    - object_name: Name of the object being dropped (used in instructions and folder names).
    Returns:
    - A list of dictionaries, each containing the coordinates for the cup dropping tasks, the instructions for each task, and the folder names for storing the data.
    """
    
    dataset = []
    if radial_bounds is None:
        radial_bounds = full_radial_bounds
    
    for i in range(n):
        next_position = generate_random_block_placements(1, 0.15, radial_bounds, avoid_positions=cup_position[:2], height=0.01)[0][:3]
        # next_position[2] += height_adjust(next_position[:2])
        
        dataset.append({
            "Coordinates": np.array(set_coords(drop_off_position, next_position, [math.atan2(drop_off_position[1], drop_off_position[0]), pi, 0], [math.atan2(next_position[1], next_position[0]), pi, 0], [0.2])),
            "Instruction": f"put the {object_name} in a random position",
            "Folder": f"{object_name}_place_random_{i+folder_start}"
        })
        dataset.append({
            "Coordinates": np.array(set_coords_cup(next_position, cup_position, [math.atan2(next_position[1], next_position[0]), pi, 0], [math.atan2(cup_position[1], cup_position[0]), pi/2,0], [0.2])),
            "Instruction": f"drop the {object_name} into the cup",
            "Folder": f"{object_name}_drop_cup_random_{i+folder_start}"
        })

    
    return dataset

def create_trash_drop_dataset(n=5, folder_start=0, cup_position=[0.2, 0], drop_off_position=[0.15,0], radial_bounds=None, object_name="trash", object_count=3):
    """
    Creates a dataset of random trash dropping tasks. The dataset includes coordinates for picking up an object, moving it to a random location, and dropping it into a cup, along with instructions for each task and folder names for storing the data.
    Parameters:
    - n: Number of random trash dropping tasks to generate.
    - folder_start: Starting index for folder naming.
    - cup_position: Position of the cup where the object will be dropped.
    - drop_off_position: Position where the object will be picked up from.
    - radial_bounds: Array specifying the radial bounds for random object placement.
    - object_name: Name of the object being dropped (used in instructions and folder names).
    - object_count: Number of objects to be dropped in each task.
    Returns:
    - A list of dictionaries, each containing the coordinates for the trash dropping tasks, the instructions for each task, and the folder names for storing the data.
    """
    
    dataset = []
    if radial_bounds is None:
        radial_bounds = full_radial_bounds
    
    for i in range(n):
        next_position = generate_random_block_placements(3, 0.15, radial_bounds, avoid_positions=cup_position[:2], height=0.01)[0][:3]
        
        dataset.append({
            "Coordinates": np.array(set_coords(drop_off_position, next_position, [math.atan2(drop_off_position[0][1], drop_off_position[0][0]), 3*pi/4, 0], [math.atan2(next_position[1], next_position[0]), 3*pi/4, 0], [0.2]) +
                                    set_coords(drop_off_position, next_position, [math.atan2(drop_off_position[1][1], drop_off_position[1][0]), 3*pi/4, 0], [math.atan2(next_position[1], next_position[0]), 3*pi/4, 0], [0.2]) +
                                    set_coords(drop_off_position, next_position, [math.atan2(drop_off_position[2][1], drop_off_position[2][0]), 3*pi/4, 0], [math.atan2(next_position[1], next_position[0]), 3*pi/4, 0], [0.2])),
            "Instruction": f"put the {object_name} in a random position",
            "Folder": f"{object_name}_place_random_{i+folder_start}"
        })
        dataset.append({
            "Coordinates": np.array(set_coords_cup(next_position, cup_position, [math.atan2(next_position[1], next_position[0]), 3*pi/4, 0], [math.atan2(cup_position[1], cup_position[0]), 3*pi/4,0], [0.2]) + 
                                    set_coords_cup(next_position, cup_position, [math.atan2(next_position[1], next_position[0]), 3*pi/4, 0], [math.atan2(cup_position[1], cup_position[0]), 3*pi/4,0], [0.2]) +
                                    set_coords_cup(next_position, cup_position, [math.atan2(next_position[1], next_position[0]), 3*pi/4, 0], [math.atan2(cup_position[1], cup_position[0]), 3*pi/4,0], [0.2])),
            "Instruction": f"drop the {object_name} into the cup",
            "Folder": f"{object_name}_drop_cup_random_{i+folder_start}"
        })

    
    return dataset

from matplotlib import pyplot as plt

def get_norm_orientation(pos):
    angle = math.atan2(pos[1], pos[0])

def create_uniform_tasks(folder_start=0,object_name="ball", grip=0.2, height=0.01):
    """
    Creates a dataset of uniform tasks for a given object. The tasks involve moving the object to specific locations around the robot, with instructions and folder names for each task.
    Parameters:
    - folder_start: Starting index for folder naming.
    - object_name: Name of the object being moved (used in instructions and folder names).
    - grip: Grip strength for the tasks.
    - height: Height of the object.
    Returns:
    - A list of dictionaries, each containing the coordinates for the movement, the instruction for the movement, and the folder name for storing the data.
    """
    radial_bounds = np.array([[0.1, -pi/2], [0.3, pi/2]])
    dataset = []
    angles = np.linspace(-pi/2, pi/2, 7, endpoint=True)
    angles_language = [3, 2, 1, 12, 11, 10, 9]
    destinations = radial_bounds[1,0] * np.vstack([np.cos(angles), np.sin(angles)]).T
    destinations = np.hstack([destinations, (height + height_adjust(destinations)) * np.ones((destinations.shape[0], 1))])
    destinations[:,0] = np.clip(destinations[:,0], 0.001, 1)
    print(destinations)
    magnitudes = np.linspace(radial_bounds[0,0], radial_bounds[1,0], 10, endpoint=False,axis=0)
    print(magnitudes.reshape(-1,1))
    sources = np.array([])
    sources_x =  (np.cos(angles).reshape(-1, 1) @ magnitudes.reshape(-1, 1).T).flatten()
    sources_x = np.clip(sources_x, 0.001, 1)
    sources_y = (np.sin(angles).reshape(-1, 1) @ magnitudes.reshape(-1, 1).T).flatten()
    sources = np.vstack([sources_x, sources_y]).T
    sources = np.hstack([sources, (height + height_adjust(sources)) * np.ones((sources.shape[0], 1))])
    for i in range(folder_start, len(sources)-1):
        start = sources[i].tolist()
        dest_index = random.randint(0, len(destinations) - 1)
        destination = destinations[dest_index].tolist()
        next_pos = sources[i+1].tolist()
        
        dataset.append({
            "Coordinates": np.array(set_coords(start, destination, [math.atan2(start[1], start[0]), 3*pi/4, 0], [math.atan2(destination[1], destination[0]), 3*pi/4, 0], [grip])),
            "Instruction": f"put the {object_name} {int(radial_bounds[1,0] * 100)} cms away from the robot at {angles_language[dest_index]} o'clock",
            "Folder": f"{object_name}_{i}_to_{dest_index}"
        })
        dataset.append({
            "Coordinates": np.array(set_coords(destination, next_pos, [math.atan2(destination[1], destination[0]), 3*pi/4, 0], [math.atan2(next_pos[1], next_pos[0]), 3*pi/4, 0], [grip])),
            "Instruction": f"put the {object_name} {int(magnitudes[i%len(magnitudes)] * 100)} cms away from the robot at {angles_language[i//len(magnitudes)]} o'clock",
            "Folder": f"{object_name}_return_{dest_index}_to_{i+1}"
        })

    return dataset

def generate_line_sample(start:np.ndarray, end:np.ndarray, grip):
    """
    Generates a list of coordinates for a linear movement from a start position to an end position, with a specified grip strength.
    Parameters:
        start (np.ndarray): The starting position.
        end (np.ndarray): The ending position.
        grip (float): The grip strength.
    Returns:
        list: A list of coordinates for the linear movement.
    """
    distance = np.linalg.norm(start[:3] - end[:3])
    num_points = math.ceil(distance / smooth_step) + 1
    points = np.linspace(start, end, num_points, endpoint=True)
    coordinates = np.hstack([points, np.ones((points.shape[0],1)) * grip])
    # t = np.linspace(0, 1, num_points, endpoint=True)
    # points = [start + (end - start) * ti for ti in t]
    return coordinates.tolist()
    
def create_bezier_curve(start, end, c1, c2):
    """Creates a Bezier curve between a start and end point with two control points, and interpolates points along the curve at regular intervals defined by smooth_step.
    Parameters:
        start (np.ndarray): The starting position.
        end (np.ndarray): The ending position.
        c1 (list): The first control point.
        c2 (list): The second control point.
    Returns:
        np.ndarray: An array of interpolated points along the Bezier curve.
    """
    dist = smooth_step
    points = np.array([start, c1, c2, end])
    curve = Bezier.Curve(np.arange(0, 1, 0.01), points)
    dists = np.linalg.norm(curve[1:, :] - curve[:-1,:], axis=1)
    interpolated = np.zeros((math.ceil(np.sum(dists)/dist) + 1, 3))
    num_points = math.ceil(np.sum(dists)/dist) + 1
    dist_index = 0
    cum_sum = 0
    interpolated[0] = start
    for i in range(1, interpolated.shape[0]):
        while cum_sum < dist and dist_index < len(dists) - 1:
            cum_sum += dists[dist_index]
            dist_index += 1
        vector = curve[dist_index] - curve[dist_index - 1]
        # print(( vector / np.linalg.norm(vector)) * (cum_sum - dist))
        point = curve[dist_index] - ( vector / np.linalg.norm(vector)) * (cum_sum - dist)
        # point = curve[dist_index]
        interpolated[i] = point
        cum_sum = cum_sum - dist
    interpolated[-1] = end
    # print([[orientation] + 0.25] * interpolated.shape[0])
    # ori = np.arange(orientation + [0.25])
    # orientations = np.linspace(orientation_start, orientation_end, interpolated.shape[0], endpoint=True)
    # result = np.hstack([interpolated, orientations, np.ones((25,1))])
    return interpolated


def generate_curve_sample(start, end, grip):
    """Generates a list of coordinates for a curved movement from a start position to an end position, with a specified grip strength. The curve is created using Bezier interpolation, and control points are calculated based on the start and end positions to ensure a smooth trajectory.
    Parameters:
        start (np.ndarray): The starting position.
        end (np.ndarray): The ending position.
        grip (float): The grip strength.
    Returns:
        list: A list of coordinates for the curved movement.
    """
    coordinates = []
    c1 = list(start[:3])
    c2 = list(end[:3])
    if start[0] < 0.1 and end[0] < 0.1 and np.sign(start[1]) != np.sign(end[1]):
        c1[0] = 0.12
        c2[0] = 0.12
    mag = np.linalg.norm(np.array(start[:3]) - np.array(end[:3]))
    c1[2] += min(mag+0.04, 0.12)
    c2[2] += min(mag+0.04, 0.12)
    # coordinates.append(c1)
    start_temp = list(start)
    start_temp[2] += 0.1
    coordinates.append(start_temp + [1])
    coordinates.append(start + [1])
    coordinates.append(start + [grip])
    curve = create_bezier_curve(start[:3], end[:3], c1, c2)
    orientations = np.linspace(start[3:6], end[3:6], curve.shape[0], endpoint=True)
    result = np.hstack([curve, orientations, np.ones((curve.shape[0],1)) * grip])
    coordinates.extend(result.tolist())
    coordinates.append(end + [grip])
    coordinates.append(end + [1])

    return coordinates

# print(generate_curve_sample([0.1, -0.1, 0.02, 0, -45, 0], [0.1, 0.1, 0.02, 0, 45, 0], 0.25))


def get_random_float(low, high):
    return random.random() * (high - low) + low

def check_bounds(position, inner_bound, outer_bound, radial_bound):
    mag = np.linalg.norm(position)
    if mag < radial_bound[0,0] or mag > radial_bound[1,0]:
        return False
    angle = math.atan2(position[1], position[0])
    if angle < -pi/2 or angle > pi/2:
        return False
    return True

def clear_buffer_frames(cap, discard_frames=20):
    for _ in range(discard_frames):
        cap.read()
    ret, frame = cap.read()
    return frame if ret else None


def create_random_dataset_smooth(n:int=20, object_name:str="block", height:float=0, grip=0.2, folder_start_num=0):
    """Creates a dataset of random tasks for a given object, where the tasks involve moving the object to random locations around the robot with smooth trajectories. The dataset includes coordinates for the movements, instructions for each task, and folder names for storing the data.
    Parameters:
    - n: Number of random tasks to generate.
    - object_name: Name of the object being moved (used in instructions and folder names).
    - height: Height of the object.
    - grip: Grip strength for the tasks.
    - folder_start_num: Starting index for folder naming.
    Returns:
    - A list of dictionaries, each containing the coordinates for the movement, the instruction for the movement, and the folder name for storing the data.
    """
    rand_samples = n
    last_orientation = [0, pi, 0]
    bounds = np.array([[0.001, -0.35],[0.35, 0.35]])
    inner_bound = np.array([[0.001, -0.1],[0.1, 0.1]])
    radial_bounds = np.array([[0.1, -pi/2],[0.3, pi/2]])
    min_dist = 0.1
    datasets = []
    height = 0.01
    last_position = [0.11, 0.0, height]
    # object_name = "block"

    for i in range(rand_samples):
        choice = random.random()
        magnitude = np.linalg.norm(last_position[:2])
        angle = math.atan2(last_position[1], last_position[0])
        # choose axis of movement
        if choice < 0.35:   # deals with x axis (forward/backward)
            if not check_bounds(np.array([last_position[0] - min_dist, last_position[1]]), inner_bound, bounds, radial_bounds):       # takes care if the current position is close to bound
                direction = "f"
            elif not check_bounds(np.array([last_position[0] + min_dist, last_position[1]]), inner_bound, bounds, radial_bounds):
                direction = "b"
            elif random.random() < 0.5:
                direction = "f"
            else:
                direction = "b"
        elif choice < 0.7:
            if pow(last_position[0] ** 2 + last_position[1] ** 2, 0.5) - min_dist < radial_bounds[0,0]:
                direction = "a"
            elif pow(last_position[0] ** 2 + last_position[1] ** 2, 0.5) + min_dist > radial_bounds[1,0]:
                direction = "t"
            elif random.random() < 0.5:
                direction = "a"
            else:
                direction = "t"
        else:
            if not check_bounds(np.array([last_position[0], last_position[1] - min_dist]), inner_bound, bounds, radial_bounds):       # takes care if the current position is close to bound
                direction = "l"
            elif not check_bounds(np.array([last_position[0], last_position[1] + min_dist]), inner_bound, bounds, radial_bounds):
                direction = "r"
            elif random.random() < 0.5:
                direction = "l"
            else:
                direction = "r"
        next_position = [0.15, 0, 0.02]
        instruction = "error"
        magnitude = np.linalg.norm(last_position[:2])
        angle = math.atan2(last_position[1], last_position[0])
        match direction:
            case "f":
                next_position = [get_random_float(last_position[0] + min_dist, math.pow(radial_bounds[1,0] ** 2 - last_position[1] ** 2, 0.5)), last_position[1], height]
                instruction = "forwards"
            case "b":
                if last_position[1] < inner_bound[1,1] and last_position[1] > inner_bound[0,1]:
                    next_position = [get_random_float( math.pow(radial_bounds[0,0] ** 2 - last_position[1] ** 2, 0.5), last_position[0] - min_dist), last_position[1], height]
                else:
                    next_position = [get_random_float(bounds[0,0], last_position[0] - min_dist), last_position[1], height]
                instruction = "backwards"
            case "a":
                f = get_random_float(magnitude+min_dist, radial_bounds[1,0])
                next_position = [f*math.cos(angle), f*math.sin(angle), height]
                instruction = "away from the robot"
            case "t":
                f = get_random_float(radial_bounds[0,0], magnitude-min_dist)
                next_position = [f*math.cos(angle), f*math.sin(angle), height]
                instruction = "towards the robot"
            case "l":
                if last_position[0] < inner_bound[1,0] and last_position[1] < 0:
                    ch = random.random()
                    if ch < 0.5:
                        next_position = [last_position[0], get_random_float(last_position[1] + min_dist, inner_bound[0,1]), height]
                    else:
                        next_position = [last_position[0], get_random_float(inner_bound[1,1], math.pow(radial_bounds[1,0] ** 2 - last_position[0] ** 2, 0.5)), height]
                else:
                    next_position = [last_position[0], get_random_float(last_position[1] + min_dist, math.pow(radial_bounds[1,0] ** 2 - last_position[0] ** 2, 0.5)), height]
                instruction = "to the left"
            case "r":
                if last_position[0] < inner_bound[1,0] and last_position[1] > 0:
                    ch = random.random()
                    if ch < 0.5:
                        next_position = [last_position[0], get_random_float(inner_bound[1,1], last_position[1] - min_dist), height]
                    else:
                        print(math.pow(radial_bounds[1,0] ** 2 - last_position[0] ** 2, 0.5))
                        next_position = [last_position[0], get_random_float(-math.pow((radial_bounds[1,0] ** 2)  - ( last_position[0] ** 2), 0.5), inner_bound[0,1]), height]
                else:
                    next_position = [last_position[0], get_random_float(-math.pow((radial_bounds[1,0] ** 2)  - ( last_position[0] ** 2), 0.5), last_position[1] - min_dist), height]
                instruction = "to the right"
        # take into account the inaccuracy of the servos
        if np.linalg.norm(next_position[:2]) > 0.21:
            next_position[2] += 0.01
        if np.linalg.norm(next_position[:2]) > 0.27:
            next_position[2] += 0.01
        # 0.3 -> 0.08
        # 0.25 -> 0.05
        coords = generate_curve_sample(last_position + [angle, pi, 0], next_position + [math.atan2(next_position[1], next_position[0]), pi, 0], grip)
        datasets.append(
            {"Coordinates": np.array(coords),
            "Instruction": f"pick up the {object_name} and move it to a random location {instruction}",
            "Folder": f"{object_name}_random_smooth_{i+folder_start_num}"
        })
        print(f"{direction}: {last_position} -> {next_position}")
        last_position = next_position
    return datasets

def place_block_smooth(start, end, orientation1, orientation2, grip):
    """Generates a list of coordinates for placing a block from a start position to an end position with smooth trajectories. The movement includes lifting the block, moving it to the end position, and placing it down, with specific orientations for each phase of the movement.
    Parameters:
    - start: The starting position of the block.
    - end: The ending position of the block.
    - orientation1: The orientation at the start position.
    - orientation2: The orientation at the end position.
    - grip: The grip strength for the tasks.
    Returns:
    - A list of coordinates for the placement movement.
    """
    result = []
    end_high = [end[0], end[1], max(end[2]+0.1, start[2]+0.1)] + orientation2
    start_high = [start[0], start[1], max(end[2]+0.1, start[2]+0.1)] + orientation1
    result.extend(generate_line_sample(np.array(start_high), np.array(start + orientation1), 1))
    result.append(start + orientation1 + [grip])
    result.extend(generate_curve_sample(start + orientation1, end_high, grip))
    result.extend(generate_line_sample(np.array(end_high), np.array(end + orientation2), grip))
    result.append(end + orientation2 + [1])
    # print(end + orientation2 + [1])
    result.extend(generate_line_sample(np.array(end + orientation2), np.array(end_high), 1))
    
    return result

def create_random_block_stack_task_smooth(block_positions):
    """
    Generates a list of coordinates for stacking blocks from their initial positions to a final stacked position with smooth trajectories. The movement includes lifting each block, moving it to the stacked position, and placing it down, with specific orientations for each phase of the movement.
    Parameters:
    - block_positions: A list of positions and orientations for each block, where each position is a list containing the x, y, z coordinates and the orientation angle.
    Returns:
    - A list of coordinates for the block stacking movement.
    """
    grip = 0.3
    task = []
    for i in range(len(block_positions) - 1):
        start_orientation = [block_positions[i][3], pi if np.linalg.norm(block_positions[i][:2]) < 0.12 else 3*pi/4, 0]
        end_orientation = [block_positions[-1][3], pi/2 if np.linalg.norm(block_positions[i][:2]) < 0.12 else 3*pi/4, 0]
        if i > 0:
            # task += generate_curve_sample(task[-1][:-1], block_positions[i][:3] + start_orientation, 1)
            pass
        task += place_block_smooth(block_positions[i][:3], block_positions[-1][:2] + [0.06 + 0.04*i + height_adjust(block_positions[-1][:2])], start_orientation, end_orientation, grip)
        
    return task


def create_random_block_stack_dataset_smooth(n=20, num_blocks=3, folder_start=0):
    """
    Creates a dataset of random block stacking tasks with smooth trajectories. The dataset includes coordinates for unstacking blocks from their initial positions and stacking them in a final position, along with instructions for each task and folder names for storing the data.
    
    Parameters:
    - n: Number of random block stacking tasks to generate.
    - num_blocks: Number of blocks in each stacking task.
    - folder_start: Starting number for folder names.
    Returns:
    - A list of dictionaries, each containing the coordinates for the block stacking tasks, the instructions for each task, and the folder names for storing the data."""
    dataset = []
    block_stack = [0.25, 0, 0.02, 0]
    radial_bounds = np.array([[0.12, -pi/2],[0.3, pi/2]])
    for i in range(n):
        block_placements = generate_random_block_placements(num_blocks, 0.12, avoid_positions=block_stack[:2], radial_bounds=radial_bounds)
        dataset.append(
            {"Coordinates": np.array(create_unstack_blocks_task(block_placements, block_stack)),
            "Instruction": f"unstack the blocks",
            "Folder": f"block_unstack_random_{i+folder_start}"
        })
        dataset.append(
            {"Coordinates": np.array(create_random_block_stack_task_smooth(block_placements)),
            "Instruction": f"stack the blocks",
            "Folder": f"block_stack_{num_blocks}_random_smooth_{i+folder_start}"
        })
        block_stack = block_placements[-1]
    
    return dataset

def create_random_drawer_dataset(n=20, num_blocks=3, folder_start=0):
    """
    Creates a dataset of random drawer tasks with smooth trajectories. The dataset includes coordinates for unstacking blocks from their initial positions and placing them in a drawer, along with instructions for each task and folder names for storing the data.
    Parameters:
    - n: Number of random drawer tasks to generate.
    - num_blocks: Number of blocks in each task.
    - folder_start: Starting number for folder names.
    Returns:
    - A list of dictionaries, each containing the coordinates for the drawer tasks, the instructions for each task, and the folder names for storing the data."""
    dataset = []
    block_stack = [0.25, 0, 0.02, 0]
    radial_bounds = np.array([[0.12, -pi/2],[0.3, -pi/8]])
    for i in range(n):
        block_placements = generate_random_block_placements(num_blocks, 0.12, avoid_positions=block_stack[:2], radial_bounds=radial_bounds)
        dataset.append(
            {"Coordinates": np.array(create_unstack_blocks_task(block_placements, block_stack)),
            "Instruction": f"unstack the blocks",
            "Folder": f"block_unstack_random_{i+folder_start}"
        })
        dataset.append(
            {"Coordinates": np.array([
                [0.18, 0.14, 0.18, pi/6, pi, 0, 0.3],
                [0.18, 0.14, 0.08, pi/6, pi, 0, 0.3],
                [0.25, 0.0, 0.08, pi/6, pi, 0, 0.3],
                [0.25, 0.0, 0.18, pi/6, pi, 0, 0.3],
            ] +
            set_coords_cup(block_placements[0][:3], [0.25,0.12, 0.2], [block_placements[0][3], 3*pi/4, 0], [pi/4, 3*pi/4, 0], [0.3]) +
            set_coords_cup(block_placements[1][:3], [0.25,0.12, 0.2], [block_placements[1][3], 3*pi/4, 0], [pi/4, 3*pi/4, 0], [0.3]) +
            set_coords_cup(block_placements[2][:3], [0.25,0.12, 0.2], [block_placements[2][3], 3*pi/4, 0], [pi/4, 3*pi/4, 0], [0.3]) +
            [
                [0.26, 0.0, 0.18, pi/6, pi, 0, 0.3],
                [0.26, 0.0, 0.08, pi/6, pi, 0, 0.3],
                [0.14, 0.16, 0.08, pi/6, pi, 0, 0.3],
            ]
            ),
            "Instruction": f"put the blocks away into the drawer",
            "Folder": f"blocks_in_drawer_random_{i+folder_start}"
        })
        # block_stack = block_placements[-1]
    
    return dataset

if __name__ == '__main__':
    # print(create_uniform_tasks())
    # print(create_random_bottle_pour_dataset(10, [0.075, -0.25], radial_bounds=full_radial_bounds, orientation1=[-pi/2, pi/2, 0], orientation2=[-pi * 5, -0.3, pi * 0.6], start_position=[0.25,0,0.12]))
    # print(generate_line_sample(np.array([0, 1, 0]), np.array([0, 2, 0]), [0]))
    # print(create_random_block_stack_dataset_smooth(1, 3, 0))
    print(np.array(generate_line_sample(np.array([0.05,-0.15,0.05, 0, pi, 0]),np.array([0.15,-0.15,0.05, 0, pi, 0]), 0.2)))
    pass