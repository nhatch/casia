# Learning letter recognition for two alphabets

This repo houses the code for a research investigation into recognizing characters with a convolutional neural network (CNN). Eventually it will involve comparing performance (learning speed and accuracy) when trained simultaneously on two alphabets vs. when trained individually on either one of them.

Currently I am merely trying to use [keras](http://keras.io/) to train a CNN that can recognize characters offline isolated characters (.gnt files) from the [CASIA dataset](http://www.nlpr.ia.ac.cn/databases/handwriting/Offline_database.html). I have included an example .gnt file in this repo, but you can download the rest [here](http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html). You probably want `HWDB1.1trn_gnt` (1873MB) and `HWDB1.1tst_gnt` (471MB).

To try it out, run this in the Python interpreter:

    import casia
    import run
    import vgg
    data_store = casia.Casia()
    model = vgg.vgg16(data_store.x_train[0].shape, len(data_store.y_train[0]))
    r = run.Run(data_store, model)
    r.run(3) # 3 epochs

## CASIA viewer

A viewer for characters from the CASIA dataset. Usage:

1. Start a web server in this directory. For example: `python -m SimpleHTTPServer`
2. Visit the server in a web browser. (In the example above, it's at [localhost:8000](http://localhost:8000).) Have fun!

## Acknowledgements

My research advisor for this project is [Greg Shakhnarovich](http://ttic.uchicago.edu/~gregory/).

The CNN architecture `vgg16` is taken from [this paper](http://arxiv.org/pdf/1409.1556.pdf).

Conversion of GB-2312 character codes into displayable strings is accomplished using `encoding.js` and `encoding-indexes.js`, both of which are taken from [inexorabletash/text-encoding](https://github.com/inexorabletash/text-encoding).

