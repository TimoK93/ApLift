"""
Data wrapper for MOT15 dataset with preprocessed data
"""

from src.datasets import MOT15


class MOT15Preprocessed(MOT15):

    def __init__(self, root_path, load_visual_embedding=False,  detector=None, **kwargs):
        super().__init__(
            root_path,
            load_visual_embedding=load_visual_embedding,
            preprocessed=True,
            **kwargs
        )
