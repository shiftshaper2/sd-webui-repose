import copy
import os, sys
import json
from math import sin, cos, pi
from PIL import Image
import numpy as np

# openpose mangling
repose_base_dir = scripts.basedir()
sys.path.insert(0, os.path.join(repose_base_dir, "..", "sd-webui-controlnet"))
from annotator.openpose import OpenposeDetector, draw_poses, decode_json_as_poses
del sys.path[0]


def union(*args):
    acc = dict()
    for a in args:
        for key in a:
            try:
                acc[key] = acc[key].union(a[key])
            except KeyError:
                acc[key] = a[key]
    return acc

NOSE = {"pose_keypoints_2d":{0}}
NECK = {"pose_keypoints_2d":{1}}
RIGHT_SHOULDER = {"pose_keypoints_2d":{2}}
RIGHT_ELBOW = {"pose_keypoints_2d":{3}}
RIGHT_WRIST = {"pose_keypoints_2d":{4}}
LEFT_SHOULDER = {"pose_keypoints_2d":{5}}
LEFT_ELBOW = {"pose_keypoints_2d":{6}}
LEFT_WRIST = {"pose_keypoints_2d":{7}}
RIGHT_HIP = {"pose_keypoints_2d":{8}}
RIGHT_KNEE = {"pose_keypoints_2d":{9}}
RIGHT_ANKLE = {"pose_keypoints_2d":{10}}
LEFT_HIP = {"pose_keypoints_2d":{11}}
LEFT_KNEE = {"pose_keypoints_2d":{12}}
LEFT_ANKLE = {"pose_keypoints_2d":{13}}
RIGHT_EYE = {"pose_keypoints_2d":{14}}
LEFT_EYE = {"pose_keypoints_2d":{15}}
RIGHT_EAR = {"pose_keypoints_2d":{16}}
LEFT_EAR = {"pose_keypoints_2d":{17}}

FACE = {"face_keypoints_2d":set(range(70))}
LEFT_HAND_ONLY = {"hand_left_keypoints_2d":set(range(21))}
LEFT_HAND = union(LEFT_HAND_ONLY, LEFT_WRIST)
RIGHT_HAND_ONLY = {"hand_right_keypoints_2d":set(range(21))}
RIGHT_HAND = union(RIGHT_HAND_ONLY, RIGHT_WRIST)

HEAD = union(NOSE, NECK, RIGHT_EYE, LEFT_EYE, RIGHT_EAR, LEFT_EAR, FACE)

LEFT_BODY = union(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, LEFT_HAND_ONLY)
RIGHT_BODY = union(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, RIGHT_HAND_ONLY)

def load_json(fpath):
    with open(fpath, 'r') as file:
        data = json.load(file)
    return data

def person(json_data, index=0):
    return json_data["people"][index]


def duplicate(person):
    return copy.deepcopy(person)


def remove(person, tracked_points):
    for key in tracked_points:
        if key in person and person[key] is not None:
            for point in tracked_points[key]:
                person[key][point * 3 + 2] = 0


def get(person, point):
    # assert there's only one key
    key, = point
    point_id, = point[key]
    return (
        person[key][point_id * 3 + 0],
        person[key][point_id * 3 + 1]
    )

def add_t(t1, t2):
    return (t1[0] + t2[0], t1[1] + t2[1])

def sub_t(t1, t2):
    return (t1[0] - t2[0], t1[1] - t2[1])

def mult_t(t, alpha):
    return (t[0] * alpha, t[1] * alpha)

def move_by(person, points, amount):
    for key in points:
        if key in person and person[key] is not None:
            for point in points[key]:
                person[key][point * 3 + 0] += amount[0]
                person[key][point * 3 + 1] += amount[1]


def replace(target, source, points, aligner, scale_factor = 1):
    s0 = get(source, aligner)
    t0 = get(target, aligner)
    for key in points:
        if key in source and source[key] is not None:
            for point in points[key]:
                # this will probably keyerror at target[key]
                dx = (source[key][point*3+0] - s0[0]) * scale_factor
                dy = (source[key][point*3+1] - s0[1]) * scale_factor

                target[key][point * 3 + 0] = t0[0] + dx
                target[key][point * 3 + 1] = t0[1] + dy
                target[key][point * 3 + 2] = source[key][point * 3 + 2]


def rotate_around(person, points, center, radians):
    for key in points:
        if key in person and person[key] is not None:
            for point in points[key]:
                dx = person[key][point * 3 + 0] - center[0]
                dy = person[key][point * 3 + 1] - center[1]

                dxp = cos(radians) * dx - sin(radians) * dy
                dyp = sin(radians) * dx + cos(radians) * dy

                person[key][point * 3 + 0] = dxp + center[0]
                person[key][point * 3 + 1] = dyp + center[1]


# conventions
img_src_dir = "img_src"
json_src_dir = "json_src"
img_out_dir = "img_out"

opd = OpenposeDetector()

def json_from_nparray(nparray):
    pose_stacks = []
    def print_json(json_string):
        pose_stacks.append(json_string)
    opd(nparray, True, True, True, True, json_pose_callback=print_json)
    return pose_stacks[0]

def load_anim(anim_path):
    base_filename = os.path.splitext(anim_path)[0]
    output_dir = os.path.join(json_src_dir, base_filename)
    json_results = []

    try:
        os.mkdir(output_dir)
    except FileExistsError:
        # load the jsons that exist there instead
        print(f"Loading previously parsed animation '{anim_path}'")

        
        # List all files in the directory
        for filename in sorted(os.listdir(output_dir)):
            if filename.endswith('.json'):
                file_path = os.path.join(output_dir, filename)
                
                # Open and load the JSON file
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                    json_results.append(data)

        return json_results


    print(f"Loading animation '{anim_path}'", end="", flush=True)
    # Open the WebP
    anim = Image.open(os.path.join(img_src_dir, anim_path))
    frame_number = 0
    
    while True:
        try:
            anim.seek(frame_number)
            
            # Convert the frame to a numpy array
            frame_array = np.array(anim.convert('RGB'))
            if len(frame_array.shape) == 3 and frame_array.shape[2]:
                frame_array = frame_array[:,:,:3]
            
            Image.fromarray(frame_array).save("total_test.png")
            # Process the frame array with the user-defined function
            json_result = json_from_nparray(frame_array)
            
            # Determine the output filename
            output_filename = os.path.join(output_dir, f"frame_{frame_number:05}.json")
            
            # Write the result string to a file
            with open(output_filename, 'w') as output_file:
                json.dump(json_result, output_file)

            json_results.append(json_result)


            # Increment the frame number to move to the next frame
            frame_number += 1
            
            # Move to the next frame
            print(".", end="", flush=True)
            
        except EOFError:
            # Exit the loop when we have processed all frames
            break

    print(" Done.")
    return json_results


def render_frames(outname, frames, height, width, frame_time=8):
    print(f"Rendering {len(frames)} frames, {height}x{width}", end="", flush=True)

    out_dir = os.path.join(img_out_dir, outname)
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        # remove everything in that directory in case frames change
        item_count = 0
        for item in os.listdir(out_dir):
            item_path = os.path.join(out_dir, item)
            os.remove(item_path)
            item_count += 1

    pose_imgs = []

    for e, people in enumerate(frames):
        poses, _, _, _ = decode_json_as_poses({"people": people, "canvas_height": height, "canvas_width":width})
        pose_img = draw_poses(poses, height, width, draw_body=True, draw_hand=True, draw_face=True)
        pose_img = Image.fromarray(pose_img)
        pose_imgs.append(pose_img)
        pose_img.save(os.path.join(out_dir, f"frame_{e:05}.png"))
        print(".", end="", flush=True)

    # Save as GIF
    pose_imgs[0].save(os.path.join(img_out_dir, f"{outname}.gif"), save_all=True, append_images=pose_imgs[1:], duration=frame_time, loop=0)
    print(" Done.")
