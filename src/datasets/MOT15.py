"""
Data wrapper for MOT15 dataset
"""

import os

from src.datasets import MOT17


class MOT15(MOT17):
    def __init__(self, root_path, load_visual_embedding=False, preprocessed=False, detector=None, **kwargs):
        super().__init__(
            detector="DPM",
            root_path=root_path,
            load_visual_embedding=load_visual_embedding,
            preprocessed=preprocessed,
            load_data=False, **kwargs
        )

        if preprocessed:
            self.root = os.path.join(root_path, "MOT15-Preprocessed")
        else:
            self.root = os.path.join(root_path, "MOT15")

        self.detector = "ALL"
        self.load_sequences()

