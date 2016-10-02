import keras.utils.np_utils as np_utils
import sys
from keras import backend as K

class Run:
  def __init__(self, data, model_constructor):
    self.x = data[0]
    self.y = data[1]
    self.xt = data[2]
    self.yt = data[3]
    self.model = model_constructor(self.x[0].shape, len(self.y[0]))

  # setting batch_size too high can cause ENOMEM
  def run(self, nb_epoch, batch_size=16):
    self.train(nb_epoch, batch_size)
    self.test(batch_size)

  def train(self, nb_epoch, batch_size=16):
    self.model.fit(self.x, self.y, validation_split=0.125, batch_size=batch_size, nb_epoch=nb_epoch)

  def test(self, batch_size=16):
    predictions = self.model.predict_classes(self.xt, batch_size=batch_size)
    predictions_c = np_utils.to_categorical(predictions, self.model.output_shape[1])
    incorrect = filter(lambda(i, (x,y)): not all(x == y), enumerate(zip(predictions_c, self.yt)))
    print "error rate: {} / {}".format(len(incorrect), len(predictions))


  ### Debugging Tools ###

  # `img_2d` should be a 2D array of floats in [0,1].
  # Prints a rough visualization to console.
  def visualize(self, img_2d):
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

  def v(self, img_index):
    visualize(self.x[img_index][0])

  # Gives the output of the layer of `model` identified by `layer_idx`
  # when the input is `datum`.
  # Example: inspect_layer(2, 0)
  def inspect_layer(self, layer_idx, datum_idx):
    get_layer_output = K.function([self.model.layers[0].input], [self.model.layers[layer_idx].output])
    layer_output = get_layer_output([[self.x[datum_idx]]])[0]
    return layer_output

