"""
Data wrapper for MOT17 dataset with preprocessed data
"""


from src.datasets import MOT17


class MOT17Preprocessed(MOT17):

    def __init__(self, detector, root_path, load_visual_embedding=False, load_to_ram=False, **kwargs):
        super().__init__(
            detector,
            root_path,
            load_visual_embedding=load_visual_embedding,
            preprocessed=True,
            load_to_ram=load_to_ram,
            **kwargs
        )
