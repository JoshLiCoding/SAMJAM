import os
import torch
import numpy as np
import time
from tqdm import tqdm
from PIL import Image
from Object import Object
from vis import draw_masks_in_frame, draw_rels_in_frame
from mask import generate_masks_first_frame, generate_masks_next_frame
from vlms.gpt import classify_and_describe_mask, generate_rels
# from vlms.qwen import classify_and_describe_mask
from sam2.build_sam import build_sam2_video_predictor
import psutil

sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
else:
    device = torch.device("cpu")
process = psutil.Process()

def generate_first_frame(first_frame_path, out_dir, vis_dir, resized_dims):
    objs = generate_masks_first_frame(first_frame_path, out_dir)
    draw_masks_in_frame(objs, 0, resized_dims, os.path.join(vis_dir, 'frame_0_masks.jpg'))
    
    for id, obj in objs.items():
        (obj_class, desc) = classify_and_describe_mask(first_frame_path, os.path.join(out_dir, '0', f'mask_{id}.jpg'))
        print('id:', id)
        print(obj_class)
        print(desc)
        obj.set_obj_class_and_desc(obj_class, desc)
    rels = generate_rels(objs, 0, first_frame_path)
    draw_rels_in_frame(objs, rels, first_frame_path, 0, os.path.join(vis_dir, 'frame_0_rels.jpg'))
    return objs

def generate_next_frames(frames_dir, out_dir, vis_dir, first_frame_objs, resized_dims):
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    inference_state = predictor.init_state(video_path=frames_dir)

    frame_names = os.listdir(frames_dir)
    frame_names.sort(key = lambda p:int(os.path.splitext(p)[0]))
    cur_objs = first_frame_objs
    total_objs = first_frame_objs
    for cur_frame_idx in range(0, len(frame_names)-1):
        next_objs = generate_masks_next_frame(frames_dir, out_dir, vis_dir, cur_objs, total_objs, resized_dims, frame_names, cur_frame_idx, inference_state, predictor)
        cur_objs = next_objs
    predictor.reset_state(inference_state)

def main(input_dir, skip_frames):
    start_time = time.time()
    resized_width, resized_height = 0, 0

    print("Resizing images to be 3x larger")
    resized_dir = 'resized_images'
    if not os.path.exists(resized_dir):
        os.makedirs(resized_dir)
    frame_names = [p for p in os.listdir(input_dir) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]]
    frame_names.sort(key = lambda p:int(os.path.splitext(p)[0]))
    for frame_name in frame_names[::skip_frames]:
        image = Image.open(os.path.join(input_dir, frame_name))
        image = image.resize((image.size[0]*3, image.size[1]*3))
        resized_width, resized_height = image.size
        image.save(os.path.join(resized_dir, frame_name))

    out_dir = 'mask_output'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    vis_dir = 'vis_output'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    
    print("==================================================")
    print("Processing first frame")
    first_frame_objs = generate_first_frame(os.path.join(resized_dir, frame_names[0]), out_dir, vis_dir, (resized_height, resized_width))
    print("memory usage: ", process.memory_info().rss / 1024 ** 2)

    print("==================================================")

    print("Processing next frames")
    generate_next_frames(resized_dir, out_dir, vis_dir, first_frame_objs, (resized_height, resized_width))
    print("memory usage: ", process.memory_info().rss / 1024 ** 2)
    print("==================================================")

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__=="__main__":
    input_dir = 'input/epic_move_plate_100_frames/'
    skip_frames = 20
    main(input_dir, skip_frames)