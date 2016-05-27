import struct
import scipy.misc
import numpy
import keras.utils.np_utils as np_utils

NB_EXAMPLES = 64
SIDE = 224 # must be a multiple of 32 to work with maxpooling in vgg16

def load(side=SIDE):
  filename = "example.gnt"
  f = open(filename, "rb")
  X_train = numpy.zeros((NB_EXAMPLES, 1, side, side), dtype=float)
  Y_train = numpy.zeros((NB_EXAMPLES,), dtype="uint16")
  for i in range(NB_EXAMPLES):
    packed_length = f.read(4)
    if packed_length == '':
      break
    length = struct.unpack("<I", packed_length)[0]
    label = struct.unpack(">H", f.read(2))[0]
    width = struct.unpack("<H", f.read(2))[0]
    height = struct.unpack("<H", f.read(2))[0]
    bytes = struct.unpack("{}B".format(height*width), f.read(height*width))
    image = numpy.array(bytes).reshape(height, width)
    image = scipy.misc.imresize(image, (side, side))
    image = (image.astype(float) / 256) - 0.5 # normalize to [-0.5,0.5] to avoid saturation
    # TODO: should also invert image so convolutional zero-padding doesn't add a "border"?
    X_train[i, 0, :, :] = image
    Y_train[i] = label
  # categorical_crossentropy loss requires the labels to be binary vectors, not integers.
  # See https://github.com/fchollet/keras/blob/master/keras/utils/np_utils.py#L8
  # and http://stackoverflow.com/questions/31997366/python-keras-shape-mismatch-error
  Y_train -= min(Y_train) # CASIA labels start at 0xb0a1
  Y_train = np_utils.to_categorical(Y_train)
  return X_train, Y_train

