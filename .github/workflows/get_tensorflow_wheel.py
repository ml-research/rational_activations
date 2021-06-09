if __name__ == '__main__':
    import sys
    python_version = str(sys.version)
    if python_version.startswith('3.6'):
        # print('Inferred Python 3.6')
        wheel = 'https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-2.4.0-cp36-cp36m-manylinux2010_x86_64.whl'
    elif python_version.startswith('3.8'):
        # print('Inferred Python 3.8')
        wheel = 'https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-2.4.0-cp38-cp38-manylinux2010_x86_64.whl'
    print(wheel)

