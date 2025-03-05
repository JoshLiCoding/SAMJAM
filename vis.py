import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

random_colors = np.random.rand(1000, 3) # assume max 1000 unique masks

def draw_masks_in_frame(objs, frame_idx, resized_dims, vis_path):
    vis = np.zeros((resized_dims[0], resized_dims[1], 3), np.uint8)
    attrs = []
    for id, obj in objs.items():
        if frame_idx not in obj.frames:
            continue
        attrs.append({'id': id, 'seg': obj.frames[frame_idx]})
    attrs.sort(key=lambda attr: (attr['seg']['bbox'][2]-attr['seg']['bbox'][0])*(attr['seg']['bbox'][3]-attr['seg']['bbox'][1]))
    for attr in reversed(attrs):
        color = random_colors[attr['id']]*255
        vis[attr['seg']['seg']] = color
    plt.imsave(vis_path, vis)
    plt.close()

def denormalize(x1, y1, x2, y2, resized_dims):
    x1 = x1 / 1000 * resized_dims[1]
    y1 = y1 / 1000 * resized_dims[0]
    x2 = x2 / 1000 * resized_dims[1]
    y2 = y2 / 1000 * resized_dims[0]
    return x1, y1, x2, y2

def draw_vlm_bbox(frame_sg, image_path, vis_path, resized_dims):
    image = Image.open(image_path)
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(image)
    for obj in frame_sg['objects']:
        color = np.random.random(3)
        x1, y1, x2, y2 = denormalize(obj['bbox'][1], obj['bbox'][0], obj['bbox'][3], obj['bbox'][2], resized_dims)
        mx = (x1+x2)/2
        my = (y1+y2)/2
        ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, edgecolor=color, facecolor='none', linewidth=3))
        text = ax.text(mx, my, f"{obj['name']} ({obj['id']})", color='black', fontsize=12)
        text.set_bbox(dict(facecolor=color, alpha=0.5, linewidth=0))
    for rel in frame_sg['relationships']:
        id_1 = rel['subj_id']
        id_2 = rel['obj_id']
        for i, obj in enumerate(frame_sg['objects']):
            if obj['id'] == id_1:
                id_1 = i
            if obj['id'] == id_2:
                id_2 = i

        color = np.random.random(3)
        bbox1 = frame_sg['objects'][id_1]['bbox']
        x11, y11, x12, y12 = denormalize(bbox1[1], bbox1[0], bbox1[3], bbox1[2], resized_dims)
        mx1 = (x11 + x12)/2
        my1 = (y11 + y12)/2
        bbox2 = frame_sg['objects'][id_2]['bbox']
        x21, y21, x22, y22 = denormalize(bbox2[1], bbox2[0], bbox2[3], bbox2[2], resized_dims)
        mx2 = (x21 + x22)/2
        my2 = (y21 + y22)/2

        ax.arrow(mx1, my1, mx2-mx1, my2-my1, width=3, head_width=25, head_length=15, facecolor=color, edgecolor=color)
        text = ax.text((mx1+mx2)/2, (my1+my2)/2, rel['predicate'], color='black', fontsize=12)
        text.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
    plt.axis('off')
    plt.savefig(vis_path, dpi='figure')
    plt.close()

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

        ax.add_patch(Rectangle(bbox[0:2], bbox[2]-bbox[0], bbox[3]-bbox[1], edgecolor=color, facecolor='none', linewidth=3))
        text = ax.text(mx, my, f"{obj.name} ({id})", color='black', fontsize=12)
        text.set_bbox(dict(facecolor=color, alpha=0.5, linewidth=0))
    for obj_pair, rel in rels.items():
        id_1, id_2 = obj_pair.split(',')

        color = np.random.random(3)
        bbox1 = objs[int(id_1)].frames[frame_idx]['bbox']
        mx1 = (bbox1[0]+bbox1[2])/2
        my1 = (bbox1[1]+bbox1[3])/2
        bbox2 = objs[int(id_2)].frames[frame_idx]['bbox']
        mx2 = (bbox2[0]+bbox2[2])/2
        my2 = (bbox2[1]+bbox2[3])/2

        ax.arrow(mx1, my1, mx2-mx1, my2-my1, width=3, head_width=25, head_length=15, facecolor=color, edgecolor=color)
        text = ax.text((mx1+mx2)/2, (my1+my2)/2, rel, color='black', fontsize=12)
        text.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
    plt.axis('off')
    plt.savefig(vis_path, dpi='figure')
    plt.close()