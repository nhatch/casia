import struct
import scipy.misc
import numpy
import keras.utils.np_utils as np_utils
import glob

NB_TRAIN_EXAMPLES = 5000
NB_TEST_EXAMPLES = 1000
SIDE = 224 # must be a multiple of 32 to work with maxpooling in vgg16
TARGET_NUM_CLASSES = 64

class Casia:
  def __init__(self, side=SIDE):
    self.side = side
    self.x_train = numpy.zeros((NB_TRAIN_EXAMPLES, 1, self.side, self.side), dtype=float)
    self.y_train = numpy.zeros((NB_TRAIN_EXAMPLES,), dtype="uint16")
    self.x_test = numpy.zeros((NB_TEST_EXAMPLES, 1, self.side, self.side), dtype=float)
    self.y_test = numpy.zeros((NB_TEST_EXAMPLES,), dtype="uint16")
    self.labels = []
    self.gnts = iter(glob.glob("gnts/*.gnt"))
    self.file = open(self.gnts.next(), "rb")
    self.load(self.x_train, self.y_train, NB_TRAIN_EXAMPLES)
    self.load(self.x_test, self.y_test, NB_TEST_EXAMPLES)
    self.file.close()
    # categorical_crossentropy loss requires the labels to be binary vectors, not integers.
    # See https://github.com/fchollet/keras/blob/master/keras/utils/np_utils.py#L8
    # and http://stackoverflow.com/questions/31997366/python-keras-shape-mismatch-error
    self.y_train -= min(self.y_train) # CASIA labels start at 0xb0a1
    self.y_train = np_utils.to_categorical(self.y_train)
    self.y_test -= min(self.y_test)
    self.y_test = np_utils.to_categorical(self.y_test)

  def load(self, x, y, num_to_load):
    index = 0
    while index < num_to_load:
      if self.read_example(x, y, index):
        index += 1

  def read_example(self, x, y, index):
    packed_length = self.file.read(4)
    if packed_length == '':
      self.file.close()
      self.file = open(self.gnts.next(), "rb")
      packed_length = self.file.read(4)
    length = struct.unpack("<I", packed_length)[0]
    label = struct.unpack(">H", self.file.read(2))[0]
    width = struct.unpack("<H", self.file.read(2))[0]
    height = struct.unpack("<H", self.file.read(2))[0]
    bytes = struct.unpack("{}B".format(height*width), self.file.read(height*width))
    if len(self.labels) < TARGET_NUM_CLASSES:
      self.labels.append(label)
    else:
      if not label in self.labels:
        return False
    image = numpy.array(bytes).reshape(height, width)
    image = scipy.misc.imresize(image, (self.side, self.side))
    image = (image.astype(float) / 256) - 0.5 # normalize to [-0.5,0.5] to avoid saturation
    # TODO: should also invert image so convolutional zero-padding doesn't add a "border"?
    x[index, 0, :, :] = image
    y[index] = label
    return True

