import keras.utils.np_utils as np_utils
import sys
from keras import backend as K
import keras.callbacks

# setting batch_size too high can cause ENOMEM
BATCH_SIZE=32

class AnnealLearningRate(keras.callbacks.Callback):
  def __init__(self, runner):
    self.runner = runner
    self.val_loss = []

  def on_epoch_end(self, epoch, logs):
    self.val_loss.append(logs['val_loss'])
    if len(self.val_loss) > 1 and self.val_loss[-1] >= self.val_loss[-2]:
      new_lr = self.runner.model.optimizer.lr.get_value() / 2
      print "decreasing learning rate to {}".format(new_lr)
      self.runner.model.optimizer.lr.set_value(new_lr)

class Runner:
  def __init__(self, data, model_constructor):
    self.data = data
    self.model = model_constructor(data.train.x[0].shape, len(data.train.y[0]))
    self.anneal_lr = AnnealLearningRate(self)

  def run(self, nb_epoch, batch_size=BATCH_SIZE):
    self.train(nb_epoch, batch_size)
    self.test(batch_size)

  def train(self, nb_epoch, batch_size=BATCH_SIZE):
    self.model.fit(self.data.train.x, self.data.train.y,
                         validation_data=(self.data.validate.x, self.data.validate.y),
                         batch_size=batch_size, nb_epoch=nb_epoch,
                         callbacks=[self.anneal_lr])

  def test(self, batch_size=BATCH_SIZE):
    predictions = self.model.predict_classes(self.data.test.x, batch_size=batch_size)
    predictions_c = np_utils.to_categorical(predictions, self.model.output_shape[1])
    incorrect = filter(lambda(i, (x,y)): not all(x == y), enumerate(zip(predictions_c, self.data.test.y)))
    print "error rate: {} / {}".format(len(incorrect), len(predictions))


  ### Debugging Tools ###

  # `img_2d` should be a 2D array of floats in [0,1].
  # Prints a rough visualization to console.
  def visualize(self, img_2d):
    for row in img_2d:
      for cell in row:
        if cell < 0.0:
          v = "##"
        elif cell < 0.3:
          v = ".."
        else:
          v = "  "
        sys.stdout.write(v)
      print

  def v(self, img_index):
    self.visualize(self.data.train.x[img_index][0])

  # Gives the output of the layer of the model identified by `layer_idx`
  # when the input is the example identified by `datum_idx`.
  # Example: inspect_layer(2, 0)
  def inspect_layer(self, layer_idx, datum_idx):
    get_layer_output = K.function([self.model.layers[0].input], [self.model.layers[layer_idx].output])
    layer_output = get_layer_output([[self.data.train.x[datum_idx]]])[0]
    return layer_output

