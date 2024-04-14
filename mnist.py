import array
import sys
import numpy

def load_images(filename):
    f = open(filename, 'rb')

    sig = f.read(4)
    dim1 = int.from_bytes(f.read(4), byteorder='big', signed=False)
    dim2 = int.from_bytes(f.read(4), byteorder='big', signed=False)
    dim3 = int.from_bytes(f.read(4), byteorder='big', signed=False)

    data = numpy.array(array.array('B', f.read()))
    result = data.reshape(dim1, dim2, dim3)

    f.close()

    return result


def load_labels(filename):
    f = open(filename, 'rb')

    sig = f.read(4)
    dim1 = int.from_bytes(f.read(4), byteorder='big', signed=False)

    result = numpy.array(array.array('B', f.read()))

    f.close()

    return result

