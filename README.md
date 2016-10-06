# Learning letter recognition for two alphabets

This repo houses the code for a research investigation into recognizing characters with a convolutional neural network (CNN). Eventually it will involve comparing performance (learning speed and accuracy) when trained simultaneously on two alphabets vs. when trained individually on either one of them.

Currently I am trying to use [keras](http://keras.io/) to train a CNN that can recognize offline isolated characters (.gnt files) from the [CASIA dataset](http://www.nlpr.ia.ac.cn/databases/handwriting/Offline_database.html). I have included an example .gnt file in this repo, but you can download the rest [here](http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html). Specifically:

http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1trn_gnt.zip (\*)
http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1tst_gnt.zip
http://www.nlpr.ia.ac.cn/databases/Download/competition/competition-gnt.zip

(\*) **Note:** I have not been able to decompress this particular file. It is very large, it seems to require ALZip to decompress it, and when I use ALZip to decompress it, there is an error about a CRC failure. For this project, I've pooled the latter two datasets and reserved one-fifth of each character's examples as a test dataset.

## Try it out

See `provision_gpu.sh`, `setup_gpu.sh`, and `extract_gnts.sh` for information about setting up the GPU.

After the GPU is set up, you will need to preprocess the data. In the following example, we've chosen to generate preprocessed data for only eight of the CASIA character classes. (Trying to do much more than eight runs into memory problems when building the model on the GPU. I'm trying to figure out how to fix that.)

    import casia
    c = casia.Casia().read_all_examples(8)
    c.save()

Then build and train a model to recognize these characters.

    import casia, run, models
    r = run.Runner(casia.Casia().load(8), models.vgg16)
    r.run(15) # 15 epochs. It takes a while for vgg16 to converge.

Some benchmarks for this dataset: Table 3 of [this paper](http://www.nlpr.ia.ac.cn/events/CHRcompetition2013/competition/ICDAR%202013%20CHR%20competition.pdf).

## CASIA viewer

A viewer for characters from the CASIA dataset. Usage:

1. Start a web server in this directory. For example: `python -m SimpleHTTPServer`
2. Visit the server in a web browser. (In the example above, it's at [localhost:8000](http://localhost:8000).) Have fun!

## Acknowledgements

My research advisor for this project is [Greg Shakhnarovich](http://ttic.uchicago.edu/~gregory/).

The CNN architecture `vgg16` is taken from [this paper](http://arxiv.org/pdf/1409.1556.pdf).

Conversion of GB-2312 character codes into displayable strings is accomplished using `encoding.js` and `encoding-indexes.js`, both of which are taken from [inexorabletash/text-encoding](https://github.com/inexorabletash/text-encoding).

