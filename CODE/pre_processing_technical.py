import os
import pandas as pd
from CODE import constants


class pre_processing_technical:
    def __init__(self):
        pass


    def to_csv(self):
        csv_file = pd.read_csv(f'{constants.VALID_DIR}/val_annotations.txt',
        sep='\t',
        header=None,
        names=['File', 'Class', 'X', 'Y', 'H', 'W'])
        return csv_file