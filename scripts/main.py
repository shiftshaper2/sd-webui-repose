import gradio as gr
import os
import time
import traceback
import shutil
from modules import script_callbacks, scripts



import copy
import sys
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


def scale(person, points, scale_factor):
    for key in points:
        if key in person and person[key] is not None:
            for point in points[key]:
                person[key][point * 3 + 0] *= scale_factor
                person[key][point * 3 + 1] *= scale_factor


def smooth(person_time, points, anchor, factor = 0.1):
    new_anchor = get(person_time[0], anchor)
    for moment in person_time:
        old_anchor = get(moment, anchor)
        new_anchor = add_t(mult_t(new_anchor, 1 - factor), mult_t(old_anchor, factor))
        move_by(moment, points, sub_t(new_anchor, old_anchor))


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
scripts_dir = "reposes"

opd = OpenposeDetector()

def json_from_nparray(nparray):
    pose_stacks = []
    def print_json(json_string):
        pose_stacks.append(json_string)
    opd(nparray, True, True, True, True, json_pose_callback=print_json)
    return pose_stacks[0]


def load_anim(anim_name):

    base_filename = anim_name
    output_dir = os.path.join(repose_base_dir, json_src_dir, base_filename)
    json_results = []
    
    # List all [JSON] files in the directory
    for filename in sorted(os.listdir(output_dir)):
        if filename.endswith('.json'):
            file_path = os.path.join(output_dir, filename)
            
            # Open and load the JSON file
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
                json_results.append(data)

    return json_results


output_queue = []


def render_frames(outname, frames, height, width, frame_time=8):
    out_dir = os.path.join(repose_base_dir, img_out_dir, outname)
    # print(f"Rendering {len(frames)} frames to '{out_dir}', {height}x{width}", end="", flush=True)

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
        # print(".", end="", flush=True)

    # Save as GIF
    anim_path = os.path.join(repose_base_dir, img_out_dir, f"{outname}.gif")
    pose_imgs[0].save(anim_path, save_all=True, append_images=pose_imgs[1:], duration=frame_time, loop=0)
    
    output_queue.append({"frames_path":out_dir,"anim":anim_path})
    print(f"Rendered {len(frames)} frames to '{os.path.abspath(out_dir)}', {height}x{width}")

    
    #print(" Done.")

# upon load, clear the log box

logbox_path = os.path.join(repose_base_dir, "logbox.log")
with open(logbox_path, 'w'):
    pass

def exec_codebox(codebox):
    
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = open(logbox_path, 'w')
    try:
        exec(codebox, dict(globals()))
    except Exception as e:
        traceback.print_exception(e)
    sys.stdout, sys.stderr = stdout, stderr

    anims = [o["anim"] for o in output_queue]
    frames = [
        os.path.join(o["frames_path"], f) 
        for o in output_queue 
        for f in os.listdir(o["frames_path"])
    ]
    
    with open(logbox_path, 'r') as logbox_file:
        logbox_result = logbox_file.read()
    return (anims, frames, logbox_result)
    

def refresh_input_gallery():
    input_gallery_root = os.path.join(repose_base_dir, img_src_dir)
    return sorted([os.path.join(input_gallery_root, f) for f in os.listdir(input_gallery_root)])

def upload_anim(fileobj, progress=gr.Progress()):

    original_filename = os.path.basename(fileobj.name)

    base_filename = os.path.basename(fileobj.name)
    anim_path = fileobj.name
    saved_filepath = os.path.join(repose_base_dir, img_src_dir, base_filename)

    shutil.copyfile(fileobj.name, saved_filepath)

    output_dir = os.path.join(repose_base_dir, json_src_dir, base_filename)
    json_results = []

    os.mkdir(output_dir)
    # lean into the FileExistsError later

    print(f"Loading animation '{base_filename}'", end="", flush=True)
    # Open the WebP
    anim = Image.open(saved_filepath)
    n_frames = anim.n_frames
    progress(0, n_frames)
    frame_number = 0
    
    while True:
        try:
            anim.seek(frame_number)
            
            # Convert the frame to a numpy array
            frame_array = np.array(anim.convert('RGB'))
            if len(frame_array.shape) == 3 and frame_array.shape[2]:
                frame_array = frame_array[:,:,:3]
            
            # Process the frame array with the user-defined function
            json_result = json_from_nparray(frame_array)
            
            # Determine the output filename
            output_filename = os.path.join(output_dir, f"frame_{frame_number:05}.json")
            
            # Write the result string to a file
            with open(output_filename, 'w') as output_file:
                json.dump(json_result, output_file)

            json_results.append(json_result)

            progress(frame_number, n_frames)

            # Increment the frame number to move to the next frame
            frame_number += 1
            
            # Move to the next frame
            print(".", end="", flush=True)
            
        except EOFError:
            # Exit the loop when we have processed all frames
            break

    progress(None)
    
    return None, refresh_input_gallery()


def list_scripts():
    scripts_root = os.path.join(repose_base_dir, scripts_dir)
    return sorted(os.listdir(scripts_root))

def load_script(script_name):
    scripts_root = os.path.join(repose_base_dir, scripts_dir)
    with open(os.path.join(scripts_root, script_name)) as sf:
        code = sf.read()
    return code

def save_script(script_name, codebox):
    scripts_root = os.path.join(repose_base_dir, scripts_dir)
    with open(os.path.join(scripts_root, script_name), 'w') as sf:
        sf.write(codebox)


# load the UI

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_component:
        with gr.Row():
            # Input Column
            with gr.Column(scale=1) as input_column:

                with gr.Tab("Documentation") as docs:
                    gr.HTML("<p>Documentation in progress; for now read the beginning of <a href='https://github.com/shiftshaper2/sd-webui-repose/blob/main/scripts/main.py'>this python script</a> up to about 'rotate around'.</p>")
            
                # Input Images
                with gr.Tab("Inputs") as inputs:
                    # gallery of images / mp4s
                    inputs_gallery = gr.Gallery(refresh_input_gallery())
                    gr.Button("Copy to Code", interactive = False)
                    uploader = gr.UploadButton(
                            file_types=[".gif",".webp"]
                        )

                    uploader.upload(fn = upload_anim, inputs=uploader, outputs=[uploader, inputs_gallery])

            # Code Column
            with gr.Column(scale=2) as code_column:
                with gr.Row() as file_row:
                    script_fname = gr.Dropdown(label = "Filename", allow_custom_value = True, choices = list_scripts())
                    load_button = gr.Button("Load")
                    save_button = gr.Button("Save")
                    run_button = gr.Button("Run")
                
                # logbox
                logfile = open(logbox_path, 'r')
                def read_all():
                    logfile.seek(0)
                    return logfile.read()
                logbox = gr.Textbox(value = read_all, interactive = False)
                
                # Code
                with open(os.path.join(repose_base_dir, "reposes/default.py")) as user_repose:
                    codebox = gr.Code(
                        value=user_repose.read(),
                        language='python',
                        lines=20,
                        interactive=True
                    )

            with gr.Column(scale=1) as gallery_column:
                output_directory = gr.Textbox(interactive=False)
                animated_output = gr.Gallery() # webp
                frames_output = gr.Gallery() # image
                frame_drop = gr.Slider()
            
            # hook in listeners, etc
            
            run_button.click(fn = exec_codebox, inputs = codebox, outputs = [animated_output, frames_output, logbox])
            load_button.click(fn = load_script, inputs = script_fname, outputs = codebox)
            save_button.click(fn = save_script, inputs = [script_fname, codebox])
        return [(ui_component, "RePose", "repose_tab")]

script_callbacks.on_ui_tabs(on_ui_tabs)
