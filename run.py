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
  predictions = np_utils.to_categorical(predictions, num_classes)
  nb_correct = len(filter(lambda(x,y): all(x == y), zip(predictions, Y_train)))
  accuracy = float(nb_correct) / len(Y_train)
  print "accuracy: {}".format(accuracy)


### Debugging Tools ###

# `img_2d` should be a 2D array of floats in [0,1].
# Prints a rough visualization to console.
def visualize(img_2d):
  for row in img_2d:
    for cell in row:
      if cell < 0.5:
        v = "#"
      elif cell < 0.8:
        v = "."
      else:
        v = " "
      sys.stdout.write(v)
    print

# Gives the output of the layer of `model` identified by `layer_idx`
# when the input is `datum`.
# Example: visualize_layer(model, 2, X_train[0])
def inspect_layer(model, layer_idx, datum):
  get_layer_output = K.function([model.layers[0].input], [model.layers[layer_idx].output])
  layer_output = get_layer_output([[datum]])[0]
  return layer_output

