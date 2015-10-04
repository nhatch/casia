# CASIA viewer

A viewer for offline isolated characters (.gnt files) from the [CASIA database](http://www.nlpr.ia.ac.cn/databases/handwriting/Offline_database.html). I have included an example .gnt file in this repo, but you can download the rest [here](http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html).

## Acknowledgements

Conversion of GB-2312 character codes into displayable strings is accomplished using encoding.js and encoding-indexes.js, both of which are taken from [inexorabletash/text-encoding](https://github.com/inexorabletash/text-encoding).

## Usage

1. Start a web server in this directory. For example: `python -m SimpleHTTPServer`
2. Visit the server in a web browser. (In the example above, it's at [localhost:8000](http://localhost:8000).) Have fun!

