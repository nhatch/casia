import struct
import scipy.misc
import numpy
import keras.utils.np_utils as np_utils
import glob
from collections import defaultdict

SIDE = 224 # must be a multiple of 32 to work with maxpooling in vgg16

class Dataset(object):
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def save(self, name):
    prefix = "{}_{}x{}_{}_".format(name, self.side(), self.side(), self.num_classes())
    numpy.save(prefix + "x.npy", self.x)
    numpy.save(prefix + "y.npy", self.y)

  def side(self):
    return len(self.x[0][0])

  def num_classes(self):
    return len(self.y[0])

  # Convert the input into a format usable with Keras
  # x : an array of numpy.array
  # y : an array of int
  @staticmethod
  def build(x, y):
    nb_examples = len(y)
    side = len(x[0][0])
    converted_x = numpy.zeros((nb_examples, 1, side, side), dtype=float)
    for i in range(nb_examples):
      converted_x[i, 0, :, :] = x[i]
    # categorical_crossentropy loss requires the labels to be binary vectors, not integers.
    # See https://github.com/fchollet/keras/blob/master/keras/utils/np_utils.py#L8
    # and http://stackoverflow.com/questions/31997366/python-keras-shape-mismatch-error
    converted_y = np_utils.to_categorical(y)
    return Dataset(converted_x, converted_y)

  @staticmethod
  def load(name, num_classes, side=SIDE):
    prefix = "{}_{}x{}_{}_".format(name, side, side, num_classes)
    return Dataset(numpy.load(prefix + "x.npy"), numpy.load(prefix + "y.npy"))

class Casia:
  def __init__(self, side=SIDE):
    self.side = side

  def load(self, num_classes):
    self.train = Dataset.load("train", num_classes)
    self.validate = Dataset.load("validate", num_classes)
    self.test = Dataset.load("test", num_classes)
    return self

  def save(self):
    self.train.save("train")
    self.validate.save("validate")
    self.test.save("test")

  def read_all_examples(self, num_classes, gnt_root="/run/gnts"):
    self.full_data = defaultdict(lambda: [])
    for filename in glob.glob(gnt_root + "/*.gnt"):
      print filename
      self.read_examples(filename, num_classes)
    self.build_datasets(num_classes)
    return self

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

  def build_datasets(self, num_classes):
    classes = self.full_data.items()
    classes.sort()
    classes_to_use = classes[:num_classes]
    data = [[],[],[]]
    labels = [[],[],[]]
    split_points = [0.0, 0.7, 0.8, 1.0]
    for class_label, class_data in classes_to_use:
      for i in range(len(data)):
        start_idx = int(len(class_data) * split_points[i])
        stop_idx = int(len(class_data) * split_points[i+1])
        data[i] += class_data[start_idx:stop_idx]
        labels[i] += [class_label]*(stop_idx - start_idx)
    self.train, self.validate, self.test = map(lambda(x,y): Dataset.build(x,y), zip(data, labels))
