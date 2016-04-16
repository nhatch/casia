# Learning letter recognition for two alphabets

This repo houses the code for a research investigation into recognizing characters with a convolutional neural network (CNN). Eventually it will involve comparing performance (learning speed and accuracy) when trained simultaneously on two alphabets vs. when trained individually on either one of them.

Currently I am merely trying to use [keras](http://keras.io/) to train a CNN that can recognize characters offline isolated characters (.gnt files) from the [CASIA dataset](http://www.nlpr.ia.ac.cn/databases/handwriting/Offline_database.html). I have included an example .gnt file in this repo, but you can download the rest [here](http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html).

The model is not yet learning at all, but if you want to try it, run this in the Python interpreter:

    import run
    run.run(3) # 3 epochs

## CASIA viewer

A viewer for characters from the CASIA dataset. Usage:

1. Start a web server in this directory. For example: `python -m SimpleHTTPServer`
2. Visit the server in a web browser. (In the example above, it's at [localhost:8000](http://localhost:8000).) Have fun!

## Acknowledgements

My research advisor for this project is [Greg Shakhnarovich](http://ttic.uchicago.edu/~gregory/).

The CNN architecture in `vgg.py` is taken from [this paper](http://arxiv.org/pdf/1409.1556.pdf).

Conversion of GB-2312 character codes into displayable strings is accomplished using `encoding.js` and `encoding-indexes.js`, both of which are taken from [inexorabletash/text-encoding](https://github.com/inexorabletash/text-encoding).

