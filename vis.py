import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw

def draw_masks_in_frame(objs, frame_idx, resized_dims, vis_path):
    vis = np.zeros((resized_dims[0], resized_dims[1], 3), np.uint8)
    for id, obj in objs.items():
        color = np.random.random(3)*255
        vis[obj.frames[frame_idx]['seg']] = color
    plt.imsave(vis_path, vis)

def draw_rels_in_frame(objs, rels, image_path, frame_idx, vis_path):
    image = Image.open(image_path)
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(image)

    for id, obj in objs.items():
        color = np.random.random(3)
        bbox = obj.frames[frame_idx]['bbox']
        mx = (bbox[0]+bbox[2])/2
        my = (bbox[1]+bbox[3])/2

        ax.add_patch(Rectangle(bbox[0:2], bbox[2]-bbox[0], bbox[3]-bbox[1], edgecolor=color, facecolor='none'))
        text = ax.text(mx, my, obj.obj_class, color='black', fontsize=12)
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