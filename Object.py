class Object:
    def __init__(self):
        self.obj_class = ''
        self.desc = ''
        self.frames = {}
    def set_obj_class_and_desc(self, obj_class, desc):
        self.obj_class = obj_class
        self.desc = desc
    def add_frame_seg(self, frame_id, seg):
        seg = {
            "seg": seg
        }
        self.frames[frame_id] = seg
    def add_bbox(self, frame_id, bbox):
        self.frames[frame_id]['bbox'] = bbox