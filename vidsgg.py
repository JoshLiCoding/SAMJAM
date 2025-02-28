import os
import torch
import numpy as np
import uuid
import time
import json
import re
from tqdm import tqdm
from PIL import Image
from Object import Object
from vis import draw_masks_in_frame, draw_rels_in_frame, draw_vlm_bbox
from mask import generate_masks_first_frame, get_sg_next_frame, match_objects
from vlms.gemini import generate_frame_scene_graph
from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
else:
    device = torch.device("cpu")

def save_scene_graph(sg_dir, frame_idx, objs, rels):
    with open(os.path.join(sg_dir, f'{frame_idx}_objs.json'), 'w', encoding='utf-8') as f:
        objs_list = []
        for id, obj in objs.items():
            objs_list.append(
                {'id': id, 
                'name': obj.name, 
                "is_hand": obj.is_hand,
                "is_moving": obj.is_moving,
                "is_moved": obj.is_moved
                })
        json.dump(objs_list, f, ensure_ascii=False, indent=4)
    with open(os.path.join(sg_dir, f'{frame_idx}_rels.json'), 'w', encoding='utf-8') as f:
        json.dump(rels, f, ensure_ascii=False, indent=4)

def generate_first_frame(first_frame_path, vis_dir, sg_dir, resized_dims):
    frame_sg = generate_frame_scene_graph(first_frame_path)
    objs = generate_masks_first_frame(first_frame_path)
    matched_objs, rels = match_objects(frame_sg, objs, 0, resized_dims)

    draw_vlm_bbox(frame_sg, first_frame_path, os.path.join(vis_dir, 'frame_0_vlm_bbox.jpg'), resized_dims)
    draw_masks_in_frame(objs, 0, resized_dims, os.path.join(vis_dir, 'frame_0_full_masks.jpg'))
    draw_masks_in_frame(matched_objs, 0, resized_dims, os.path.join(vis_dir, 'frame_0_matched_masks.jpg'))
    draw_rels_in_frame(matched_objs, rels, first_frame_path, 0, os.path.join(vis_dir, 'frame_0_matched_objs_rels.jpg'))
    save_scene_graph(sg_dir, 0, matched_objs, rels)
    
    return matched_objs, rels

def generate_next_frames(frames_dir, vis_dir, sg_dir, first_frame_objs, rels, resized_dims):
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    inference_state = predictor.init_state(video_path=frames_dir)

    frame_names = os.listdir(frames_dir)
    frame_names.sort(key = lambda p:int(os.path.splitext(p)[0]))
    cur_objs = first_frame_objs
    total_objs = first_frame_objs
    for cur_frame_idx in range(0, len(frame_names)-1):
        frame_sg = generate_frame_scene_graph(os.path.join(frames_dir, frame_names[cur_frame_idx+1]))
        draw_vlm_bbox(frame_sg, os.path.join(frames_dir, frame_names[cur_frame_idx+1]), os.path.join(vis_dir, f'frame_{cur_frame_idx+1}_vlm_bbox.jpg'), resized_dims)

        next_objs = get_sg_next_frame(frames_dir, vis_dir, cur_objs, frame_sg, total_objs, rels, resized_dims, frame_names, cur_frame_idx, inference_state, predictor)
        cur_objs = next_objs
        save_scene_graph(sg_dir, cur_frame_idx+1, total_objs, rels)
    
    predictor.reset_state(inference_state)
    return total_objs

def main(input_dir, skip_frames):
    start_time = time.time()
    resized_width, resized_height = 0, 0

    output_dir = os.path.join('output', str(uuid.uuid4()))
    os.makedirs(output_dir)

    print("Resizing images to be 3x larger")
    resized_dir = os.path.join(output_dir, 'resized_images')
    os.makedirs(resized_dir)
    frame_names = [p for p in os.listdir(input_dir) if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]]
    frame_names.sort(key = lambda p:int(re.sub("[^0-9]", "", os.path.splitext(p)[0])))
    cleaned_frame_names = list(map(lambda frame_name:re.sub("[^0-9]", "", os.path.splitext(frame_name)[0])+'.jpg', frame_names))
    for frame_name, cleaned_frame_name in zip(frame_names[::skip_frames], cleaned_frame_names[::skip_frames]):
        image = Image.open(os.path.join(input_dir, frame_name))
        image = image.resize((image.size[0]*3, image.size[1]*3))
        resized_width, resized_height = image.size
        image.save(os.path.join(resized_dir, cleaned_frame_name))

    vis_dir = os.path.join(output_dir, 'vis_output')
    os.makedirs(vis_dir)
    sg_dir = os.path.join(output_dir, 'scene_graph_output')
    os.makedirs(sg_dir)
    
    print("==================================================")
    print("Processing first frame")
    first_frame_objs, rels = generate_first_frame(os.path.join(resized_dir, cleaned_frame_names[0]), vis_dir, sg_dir, (resized_height, resized_width))

    print("==================================================")

    print("Processing next frames")
    objs = generate_next_frames(resized_dir, vis_dir, sg_dir, first_frame_objs, rels, (resized_height, resized_width))
    print("==================================================")

    # out_dir = 'mask_output'
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)
    
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__=="__main__":
    input_dir = 'input/epic_cut_bell_pepper_220_frames'
    skip_frames = 22
    main(input_dir, skip_frames)