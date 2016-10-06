#!/bin/bash
#
# provision_gpu.sh
#
# Copies files from one box to an EC2 instance. These files must be present on the instance in order for setup_gpu.sh to succeed and in order to run the code in this repo.
# See individual line comments to learn how to obtain the files.

set -e

HOST_ADDR=$1 # address of the EC2 instance you are setting up.
SSH_KEY=~/.ssh/aws_personal.pem # SSH key to access the EC2 instance

# cuDNN installer
# Download archive from https://developer.nvidia.com/rdp/cudnn-download. You will need to create an account and answer some questions about intended use.
# Get one of the versions of cuDNN for Linux matching the version of CUDA you will use.
# As of this writing, the latest version officially supported by Theano is cuDNN 5.0.
scp -i $SSH_KEY cudnn-7.5-linux-x64-v* ec2-user@$HOST_ADDR:~

# SSH key for git
scp -i $SSH_KEY ~/.ssh/id_rsa ec2-user@$HOST_ADDR:~/.ssh

# SSH key for other EC2 instances
# Useful for moving around serialized models to try using them on different architectures
scp -i $SSH_KEY $SSH_KEY ec2-user@$HOST_ADDR:~/.ssh

# Training and testing data for CASIA
# See README for information about how to download these. They are large.
# TODO add these files to S3 or somewhere hosted *not* in China
scp -i $SSH_KEY HWDB1.1tst_gnt.zip ec2-user@$HOST_ADDR:~
scp -i $SSH_KEY competition-gnt.zip ec2-user@$HOST_ADDR:~

# The script from this repo to set up the EC2 instance
scp -i $SSH_KEY setup_gpu.sh ec2-user@$HOST_ADDR:~

