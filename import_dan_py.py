from contextlib import contextmanager

@contextmanager
def ImportDanPy():
    try:
        yield
    except ImportError as e:
        module_name = str(e).split('No module named ', 1)[1].strip().strip('"\'')
        print(f'Missing module {module_name}. Please download at')
        print(f'https://github.com/Daniel-Chin/Python_Lib/blob/master/{module_name}.py')
        input('Press Enter to quit...')
        raise e
