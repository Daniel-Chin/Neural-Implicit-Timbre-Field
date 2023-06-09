print('loss_meta...')
import sys
import os
saved_stdout = sys.stdout
TEMP = 'temp.txt'
with open(TEMP, 'w') as f:
    sys.stdout = f
    from torchWork import *
    sys.stdout = saved_stdout
try:
    os.remove(TEMP)
except FileNotFoundError:
    pass    # if multiple processes are running

AbstractLossNode = loss_tree.AbstractLossNode

def main():
    absLossRoot = AbstractLossNode('loss_root', [
        'harmonics', 
        'dredge_regularize', 
    ])
    with open('losses.py', 'w') as f:
        loss_tree.writeCode(f, absLossRoot)

main()
