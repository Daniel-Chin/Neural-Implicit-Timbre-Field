import torch

from a import *

def loadNITF(filename):
    nitf = NITF()
    model.load_state_dict(torch.load(path.join(
        trainer_path, modelFileName(name, i, epoch), 
    ), map_location=DEVICE))
