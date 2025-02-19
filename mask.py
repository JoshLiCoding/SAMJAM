import os
import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from vis import draw_masks_in_frame, draw_rels_in_frame
import cv2
import matplotlib.pyplot as plt
from Object import Object
sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
else:
    device = torch.device("cpu")
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device)

mask_generator = SAM2AutomaticMaskGenerator(
    model=sam2,
    points_per_side=64,
    points_per_batch=64,
    stability_score_thresh=0.8,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    use_m2m=True
)
id_ctr = 0

def calculate_iou(bbox1, bbox2):
    x_min_intersect = max(bbox1[0], bbox2[0])
    y_min_intersect = max(bbox1[1], bbox2[1])
    x_max_intersect = min(bbox1[2], bbox2[2])
    y_max_intersect = min(bbox1[3], bbox2[3])
    intersection_area = max(0, x_max_intersect - x_min_intersect) * max(0, y_max_intersect - y_min_intersect)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - intersection_area
    iou = intersection_area / union_area
    return iou

def mask_to_bbox(mask):
    rows, cols = np.where(mask)
    x1 = np.min(cols)
    y1 = np.min(rows)
    x2 = np.max(cols)
    y2 = np.max(rows)
    return x1, y1, x2, y2

def match_objects(frame_sg, objs, frame_idx, resized_dims):
    id_map = {}
    matched_objs = {}
    for vlm_obj in frame_sg['objects']:
        vlm_x1 = vlm_obj['bbox'][1] / 1000 * resized_dims[1]
        vlm_y1 = vlm_obj['bbox'][0] / 1000 * resized_dims[0]
        vlm_x2 = vlm_obj['bbox'][3] / 1000 * resized_dims[1]
        vlm_y2 = vlm_obj['bbox'][2] / 1000 * resized_dims[0]

        highest_iou = 0
        best_obj_id = -1
        best_sam_obj = ''
        for id, sam_obj in objs.items():
            iou = calculate_iou([vlm_x1, vlm_y1, vlm_x2, vlm_y2], sam_obj.frames[frame_idx]['bbox'])
            if iou >= highest_iou:
                highest_iou = iou
                best_obj_id = id
                best_sam_obj = sam_obj
        if highest_iou >= 0.1:
            id_map[vlm_obj['id']] = best_obj_id
            best_sam_obj.name = vlm_obj['name']
            best_sam_obj.is_hand = vlm_obj['is_hand']
            best_sam_obj.is_moving = vlm_obj['is_moving']
            if best_sam_obj.is_moving:
                best_sam_obj.is_moved = True
            matched_objs[best_obj_id] = best_sam_obj
    
    rels = {}
    for rel in frame_sg['relationships']:
        if (rel['subj_id'] not in id_map) or (rel['obj_id'] not in id_map):
            continue
        mapped_subj_id = id_map[rel['subj_id']]
        mapped_obj_id = id_map[rel['obj_id']]
        rels[f'{mapped_subj_id},{mapped_obj_id}'] = rel['predicate']
    return (matched_objs, rels)

def generate_masks_first_frame(first_frame_path):
    frame = np.array(Image.open(first_frame_path))
    masks = mask_generator.generate(frame)

    objs = {}
    for i, mask in enumerate(masks):
        obj = Object()
        x1, y1, w, h = map(int, mask['bbox'])
        obj.add_frame_seg(0, mask['segmentation'], [x1, y1, x1+w, y1+h])
        global id_ctr
        objs[id_ctr] = obj
        id_ctr += 1
    
    return objs

def get_sg_next_frame(frames_dir, vis_dir, cur_objs, frame_sg, total_objs, rels, resized_dims, frame_names, cur_frame_idx, inference_state, predictor):
    for id, obj in cur_objs.items():
        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=cur_frame_idx,
            obj_id=id,
            mask=obj.frames[cur_frame_idx]["seg"]
        )
    
    propagated_mask_region = np.zeros((resized_dims), np.bool)
    next_objs = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        if out_frame_idx == cur_frame_idx+1:
            for i, out_obj_id in enumerate(out_obj_ids):
                out_mask = (out_mask_logits[i] > 0.0).cpu().numpy().reshape(resized_dims)
                if np.sum(out_mask) == 0:
                    continue
                out_bbox = mask_to_bbox(out_mask)
                propagated_mask_region |= out_mask
                total_objs[out_obj_id].add_frame_seg(out_frame_idx, out_mask, out_bbox)

                obj = Object()
                obj.name = total_objs[out_obj_id].name
                obj.add_frame_seg(out_frame_idx, out_mask, out_bbox)
                next_objs[out_obj_id] = obj
    draw_masks_in_frame(total_objs, cur_frame_idx+1, resized_dims, os.path.join(vis_dir, f'frame_{cur_frame_idx+1}_propagated_masks.jpg'))
    
    next_frame = np.array(Image.open(os.path.join(frames_dir, frame_names[cur_frame_idx+1])))
    generated_masks = mask_generator.generate(next_frame)
    
    for i, mask in enumerate(generated_masks):
        intersection = mask['segmentation'] & propagated_mask_region
        if np.sum(intersection) / np.sum(mask['segmentation']) < 0.5:
            global id_ctr
            obj = Object()
            x1, y1, w, h = map(int, mask['bbox'])
            obj.add_frame_seg(cur_frame_idx+1, mask['segmentation'], [x1, y1, x1+w, y1+h])
            next_objs[id_ctr] = obj
            id_ctr += 1
    
    matched_objs, next_rels = match_objects(frame_sg, next_objs, cur_frame_idx+1, resized_dims)
    next_objs_matched = {}
    for id, obj in matched_objs.items():
        if id not in total_objs:
            total_objs[id] = obj
            next_objs_matched[id] = obj
        else:
            total_objs[id].name = obj.name
            if (total_objs[id].is_moving) or (obj.is_moving):
                for key in list(rels):
                    subj_id, obj_id = key.split(',')
                    if int(id) == int(subj_id) or int(id) == int(obj_id):
                        x = rels.pop(key)
                total_objs[id].is_moving = obj.is_moving
                if obj.is_moving:
                    total_objs[id].is_moved = True
                
    for obj_subj_pair, predicate in next_rels.items():
        rels[obj_subj_pair] = predicate
    
    draw_masks_in_frame(total_objs, cur_frame_idx+1, resized_dims, os.path.join(vis_dir, f'frame_{cur_frame_idx+1}_total_masks.jpg'))
    draw_rels_in_frame(matched_objs, next_rels, os.path.join(frames_dir, frame_names[cur_frame_idx+1]), cur_frame_idx+1, os.path.join(vis_dir, f'frame_{cur_frame_idx+1}_matched_objs_rels.jpg'))
    return next_objs_matched