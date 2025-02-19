import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

def draw_masks_in_frame(objs, frame_idx, resized_dims, vis_path):
    vis = np.zeros((resized_dims[0], resized_dims[1], 3), np.uint8)
    attrs = []
    for id, obj in objs.items():
        if frame_idx not in obj.frames:
            continue
        attrs.append(obj.frames[frame_idx])
    attrs.sort(key=lambda attr: (attr['bbox'][2]-attr['bbox'][0])*(attr['bbox'][3]-attr['bbox'][1]))
    for attr in reversed(attrs):
        color = np.random.random(3)*255
        vis[attr['seg']] = color
    plt.imsave(vis_path, vis)

def draw_vlm_bbox(frame_sg, image_path, vis_path, resized_dims):
    image = Image.open(image_path)
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(image)
    for obj in frame_sg['objects']:
        color = np.random.random(3)
        x1 = obj['bbox'][1] / 1000 * resized_dims[1]
        y1 = obj['bbox'][0] / 1000 * resized_dims[0]
        x2 = obj['bbox'][3] / 1000 * resized_dims[1]
        y2 = obj['bbox'][2] / 1000 * resized_dims[0]
        ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, edgecolor=color, facecolor='none'))
    plt.axis('off')
    plt.savefig(vis_path, dpi='figure')


def draw_rels_in_frame(objs, rels, image_path, frame_idx, vis_path):
    image = Image.open(image_path)
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(image)

    for id, obj in objs.items():
        if frame_idx not in obj.frames:
            continue
        color = np.random.random(3)
        bbox = obj.frames[frame_idx]['bbox']
        mx = (bbox[0]+bbox[2])/2
        my = (bbox[1]+bbox[3])/2

        ax.add_patch(Rectangle(bbox[0:2], bbox[2]-bbox[0], bbox[3]-bbox[1], edgecolor=color, facecolor='none'))
        text = ax.text(mx, my, f"{obj.name} ({id})", color='black', fontsize=12)
        text.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
    for obj_pair, rel in rels.items():
        id_1, id_2 = obj_pair.split(',')

        color = np.random.random(3)
        bbox1 = objs[int(id_1)].frames[frame_idx]['bbox']
        mx1 = (bbox1[0]+bbox1[2])/2
        my1 = (bbox1[1]+bbox1[3])/2
        bbox2 = objs[int(id_2)].frames[frame_idx]['bbox']
        mx2 = (bbox2[0]+bbox2[2])/2
        my2 = (bbox2[1]+bbox2[3])/2

        ax.arrow(mx1, my1, mx2-mx1, my2-my1, width=3, head_width=15, head_length=10, facecolor=color, edgecolor=color)
        text = ax.text((mx1+mx2)/2, (my1+my2)/2, rel, color='black', fontsize=12)
        text.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
    plt.axis('off')
    plt.savefig(vis_path, dpi='figure')