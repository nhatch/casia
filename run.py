import keras.utils.np_utils as np_utils
import sys
from keras import backend as K

BATCH_SIZE = 16 # setting this too high causes ENOMEM on a GPU

class Run:
  def __init__(self, data_store, model):
    self.data_store = data_store
    self.model = model

  def run(self, nb_epoch):
    self.train(nb_epoch)
    self.test()

  def train(self, nb_epoch):
    self.model.fit(self.data_store.x_train, self.data_store.y_train, batch_size=BATCH_SIZE, nb_epoch=nb_epoch)

  def test(self):
    predictions = self.model.predict_classes(self.data_store.x_test, batch_size=BATCH_SIZE)
    predictions_c = np_utils.to_categorical(predictions, self.model.output_shape[1])
    incorrect = filter(lambda(i, (x,y)): not all(x == y), enumerate(zip(predictions_c, self.data_store.y_test)))
    errors = len(incorrect)
    print "error rate: {} / {}".format(errors, len(predictions))


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
    visualize(self.data_store.x_train[img_index][0])

  # Gives the output of the layer of `model` identified by `layer_idx`
  # when the input is `datum`.
  # Example: inspect_layer(2, 0)
  def inspect_layer(self, layer_idx, datum_idx):
    get_layer_output = K.function([self.model.layers[0].input], [self.model.layers[layer_idx].output])
    layer_output = get_layer_output([[self.data_store.x_train[datum_idx]]])[0]
    return layer_output

