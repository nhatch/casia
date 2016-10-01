import struct
import scipy.misc
import numpy
import keras.utils.np_utils as np_utils
import glob
from collections import defaultdict

SIDE = 224 # must be a multiple of 32 to work with maxpooling in vgg16

class Casia:
  def __init__(self, gnt_root="/run/gnts", side=SIDE):
    self.side = side
    self.full_data = defaultdict(lambda: [])
    self.gnt_root = gnt_root

  def load_data(self, num_classes):
    prefix = "casia_{}x{}_{}_".format(SIDE, SIDE, num_classes)
    return map(lambda suffix: numpy.load(prefix + suffix + ".npy"), ["x", "y", "xt", "yt"])

  def save_data(self, num_classes):
    data_with_suffixes = zip(self.data(num_classes), ["x", "y", "xt", "yt"])
    prefix = "casia_{}x{}_{}_".format(SIDE, SIDE, num_classes)
    for data, suffix in data_with_suffixes:
      numpy.save(prefix + suffix + ".npy", data)

  def read_all_examples(self, num_classes):
    for filename in glob.glob(self.gnt_root + "/*.gnt"):
      print filename
      self.read_examples(filename, num_classes)

  def read_examples(self, filename, num_classes):
    f = open(filename, "rb")
    while True:
      packed_length = f.read(4)
      if packed_length == '':
        break
      length = struct.unpack("<I", packed_length)[0]
      label = struct.unpack(">H", f.read(2))[0]
      label -= 0xb0a1 # CASIA labels start at 0xb0a1
      width = struct.unpack("<H", f.read(2))[0]
      height = struct.unpack("<H", f.read(2))[0]
      bytes = struct.unpack("{}B".format(height*width), f.read(height*width))
      existing_labels = self.full_data.keys()
      if (label in existing_labels) or (len(existing_labels) < num_classes):
        image = numpy.array(bytes).reshape(height, width)
        image = scipy.misc.imresize(image, (self.side, self.side))
        image = (image.astype(float) / 256) - 0.5 # normalize to [-0.5,0.5] to avoid saturation
        # TODO: should also invert image so convolutional zero-padding doesn't add a "border"?
        self.full_data[label].append(image)
    f.close()

  def data(self, num_classes):
    items = self.full_data.items()
    items.sort()
    items_to_use = items[:num_classes]
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for label, data in items_to_use:
      split_point = len(data) * 4 / 5
      train_data += data[:split_point]
      test_data += data[split_point:]
      train_labels += [label]*split_point
      test_labels += [label]*(len(data) - split_point)
    nb_train = len(train_labels)
    nb_test = len(test_labels)
    x_train = numpy.zeros((nb_train, 1, self.side, self.side), dtype=float)
    x_test = numpy.zeros((nb_test, 1, self.side, self.side), dtype=float)
    for i in range(nb_train):
      x_train[i, 0, :, :] = train_data[i]
    for i in range(nb_test):
      x_test[i, 0, :, :] = test_data[i]
    # categorical_crossentropy loss requires the labels to be binary vectors, not integers.
    # See https://github.com/fchollet/keras/blob/master/keras/utils/np_utils.py#L8
    # and http://stackoverflow.com/questions/31997366/python-keras-shape-mismatch-error
    y_train = np_utils.to_categorical(train_labels)
    y_test = np_utils.to_categorical(test_labels)
    return x_train, y_train, x_test, y_test
