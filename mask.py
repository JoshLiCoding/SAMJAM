import os
import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from vis import draw_masks_in_frame
import cv2
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
    points_per_batch=32,
    stability_score_thresh=0.9,
    box_nms_thresh=0.2,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=1500,
    use_m2m=True
)
id_ctr = 0

def generate_masks_first_frame(first_frame_path, out_dir):
    frame = np.array(Image.open(first_frame_path))
    masks = mask_generator.generate(frame)
    if not os.path.exists(os.path.join(out_dir, '0')):
        os.makedirs(os.path.join(out_dir, '0'))

    objs = {}
    for i, mask in enumerate(masks):
        x1, y1, w, h = map(int, mask['bbox'])
        cropped_frame = np.copy(frame[y1:y1+h, x1:x1+w])
        cropped_mask = mask['segmentation'][y1:y1+h, x1:x1+w]
        cropped_frame[~cropped_mask] = [255, 255, 255] # fill background with white space
        Image.fromarray(cropped_frame).save(os.path.join(out_dir, '0', f'mask_{i}.jpg'))

        obj = Object()
        obj.add_frame_seg(0, mask['segmentation'])
        obj.add_bbox(0, [x1, y1, x1+w, y1+h])
        global id_ctr
        objs[id_ctr] = obj
        id_ctr += 1
    
    return objs

def generate_masks_next_frame(frames_dir, out_dir, vis_dir, cur_objs, total_objs, resized_dims, frame_names, cur_frame_idx, inference_state, predictor):
    for id, obj in cur_objs.items():
        _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=cur_frame_idx,
            obj_id=id,
            mask=obj.frames[cur_frame_idx]["seg"]
        )
    
    propagated_mask_region = np.zeros((resized_dims), np.bool)
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        if out_frame_idx == cur_frame_idx+1:
            for i, out_obj_id in enumerate(out_obj_ids):
                out_mask = (out_mask_logits[i] > 0.0).cpu().numpy().reshape(resized_dims)
                propagated_mask_region |= out_mask
                total_objs[out_obj_id].add_frame_seg(out_frame_idx, out_mask)
    draw_masks_in_frame(total_objs, cur_frame_idx+1, resized_dims, os.path.join(vis_dir, f'frame_{cur_frame_idx+1}_propagated_masks.jpg'))
    
    
    next_frame = np.array(Image.open(os.path.join(frames_dir, frame_names[cur_frame_idx+1])))
    generated_masks = mask_generator.generate(next_frame)
    next_objs = {}

    if not os.path.exists(os.path.join(out_dir, str(cur_frame_idx+1))):
        os.makedirs(os.path.join(out_dir, str(cur_frame_idx+1)))
    
    for i, mask in enumerate(generated_masks):
        intersection = mask['segmentation'] & propagated_mask_region
        if sum(sum(intersection)) / sum(sum(mask['segmentation'])) < 0.5:
            global id_ctr

            x1, y1, w, h = map(int, mask['bbox'])
            cropped_frame = np.copy(next_frame[y1:y1+h, x1:x1+w])
            cropped_mask = mask['segmentation'][y1:y1+h, x1:x1+w]
            cropped_frame[~cropped_mask] = [255, 255, 255] # fill background with white space
            Image.fromarray(cropped_frame).save(os.path.join(out_dir, str(cur_frame_idx+1), f'mask_{id_ctr}.jpg'))

            obj = Object()
            obj.add_frame_seg(cur_frame_idx+1, mask['segmentation'])
            obj.add_bbox(cur_frame_idx+1, [x1, y1, x1+w, y1+h])
            next_objs[id_ctr] = obj
            total_objs[id_ctr] = obj
            id_ctr += 1

    
    draw_masks_in_frame(next_objs, cur_frame_idx+1, resized_dims, os.path.join(vis_dir, f'frame_{cur_frame_idx+1}_new_masks.jpg'))
    return next_objs