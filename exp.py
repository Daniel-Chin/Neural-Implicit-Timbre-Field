from typing import *

import a

def getTrainers(n_pages, dataset):
    trainers: List[a.Trainer] = []
    for _ in range(2):
        trainers.append(a.Trainer(
            64, 6, 2, 
            2 ** 12, n_pages, 
        ).ready(dataset))
        trainers.append(a.Trainer(
            128, 6, 2, 
            2 ** 12, n_pages, 
        ).ready(dataset))
        trainers.append(a.Trainer(
            256, 6, 2, 
            2 ** 12, n_pages, 
        ).ready(dataset))
    print('trainers ok')
    return trainers
