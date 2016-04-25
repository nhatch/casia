import casia
import vgg
import keras.utils.np_utils as np_utils
import sys
from keras import backend as K

X_train, Y_train = casia.load()
num_classes = len(Y_train[0])
model = vgg.vgg16(X_train[0].shape, num_classes)

def run(nb_epoch):
  train(nb_epoch)
  test()

def train(nb_epoch):
  model.fit(X_train, Y_train, batch_size=32, nb_epoch=nb_epoch)

def test():
  predictions = model.predict_classes(X_train, batch_size=32)
  predictions_c = np_utils.to_categorical(predictions, num_classes)
  incorrect = filter(lambda(i, (x,y)): not all(x == y), enumerate(zip(predictions_c, Y_train)))
  errors = len(incorrect)
  if errors > 0:
    for i, (x,y) in incorrect:
      print "Should have been {}, predicted {}:".format(i, predictions[i])
      print inspect_layer(-2, i)
  print "error rate: {} / {}".format(errors, len(Y_train))



### Debugging Tools ###

# `img_2d` should be a 2D array of floats in [0,1].
# Prints a rough visualization to console.
def visualize(img_2d):
  for row in img_2d:
    for cell in row:
      if cell < 0.0:
        v = "#"
      elif cell < 0.3:
        v = "."
      else:
        v = " "
      sys.stdout.write(v)
    print

def v(img_index):
  visualize(X_train[img_index][0])

# Gives the output of the layer of `model` identified by `layer_idx`
# when the input is `datum`.
# Example: inspect_layer(2, 0)
def inspect_layer(layer_idx, datum_idx):
  get_layer_output = K.function([model.layers[0].input], [model.layers[layer_idx].output])
  layer_output = get_layer_output([[X_train[datum_idx]]])[0]
  return layer_output

