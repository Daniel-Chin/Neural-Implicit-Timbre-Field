import os

op = '''
exp_mss.py
'''.strip()
# op = input('Which exp? ')

os.system(f'python train.py {op}')
