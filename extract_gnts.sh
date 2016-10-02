#!/bin/bash
# The extracted versions of these files are usually too large to fit on the
# persistent storage of an EC2 instance. This script helps re-extract them to
# larger temporary storage after the instance is started.
sudo chown ec2-user /run
mkdir /run/gnts
unzip ~/competition-gnt.zip -d /run/gnts
unzip ~/HWDB1.1tst_gnt.zip -d /run/gnts
