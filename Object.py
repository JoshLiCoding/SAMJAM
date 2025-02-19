class Object:
    def __init__(self):
        self.name = ''
        self.frames = {}
        self.is_hand = False
        self.is_interacting = False
        self.is_moved = False
    def add_frame_seg(self, frame_id, seg, bbox):
        attr = {
            "seg": seg,
            "bbox": bbox
        }
        self.frames[frame_id] = attr