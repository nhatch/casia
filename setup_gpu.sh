#!/bin/bash
#
# setup_gpu.sh
#
# This script is intended to be run on a newly created EC2 instance. This instance should be using the free Red Hat AMI and P2 or G2 hardware. It will set up the instance to the point that it can run the code in this repo.

set -e

sudo yum install gcc-c++ wget vim unzip screen

# Install CUDA
wget http://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-repo-rhel7-8.0.44-1.x86_64.rpm
sudo rpm -i cuda-repo-rhel7-* # ignore NOKEY warning
DKMS_PKG=dkms-2.2.0.3-34.git.9e0394d.el7.noarch.rpm # version might have been updated, you can find the new version by going to rpmfind.net/linux/epel/7/x86_64/d and ctrl-F for 'dkms'
wget ftp://rpmfind.net/linux/epel/7/x86_64/d/${DKMS_PKG}
sudo yum localinstall ${DKMS_PKG} --nogpgcheck
# Hopefully it installs a TensorFlow-compatible version.
# If not, try `sudo yum list cuda`, pick a version, and `sudo yum install cuda-#{version}`
sudo yum install cuda # this is a big download
# Gives the following non-fatal error
# The headers it wants are probably at /usr/src/kernels/3.10.0-327.13.1.el7.x86_64/
# Maybe a config change could fix?
#
# Error! echo
# Your kernel headers for kernel 3.10.0-327.el7.x86_64 cannot be found at
# /lib/modules/3.10.0-327.el7.x86_64/build or /lib/modules/3.10.0-327.el7.x86_64/source.
# warning: %post(nvidia-kmod-1:352.79-2.el7.x86_64) scriptlet failed, exit status 1
# Non-fatal POSTIN scriptlet failure in rpm package 1:nvidia-kmod-352.79-2.el7.x86_64

echo 'export CUDA_HOME=/usr/local/cuda/' >> .bashrc
echo 'export PATH=$PATH:${CUDA_HOME}bin' >> .bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}lib64' >> .bashrc
source .bashrc

# Verify you have an NVIDIA-compatible GPU
sudo yum install pciutils
lspci | grep -i nvidia

# Install NVIDIA driver
# If you use a more recent version of CUDA, you may need to use a more recent driver.
# Search for drivers at http://www.nvidia.com/Download/Find.aspx?lang=en-us
sudo yum install kernel-devel
wget http://us.download.nvidia.com/XFree86/Linux-x86_64/375.20/NVIDIA-Linux-x86_64-375.20.run
chmod +x NVIDIA-Linux-x86_64-*
KERNEL_SOURCE_PATH=`rpm -ql kernel-devel | head -n1`/
sudo ./NVIDIA-Linux-x86_64-* --kernel-source-path=${KERNEL_SOURCE_PATH}
# Follow prompts. Don't try to use dkms, install 32-bit compatibility library, or worry about X.
# You may also need to disable Nouveau kernel driver -- the NVIDIA install script can do this automatically with a modprobe configuration change; this requires rebooting the EC2 instance.

# Install python dependencies
wget https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
sudo yum install epel-release-latest-7.noarch.rpm
sudo yum repolist
sudo yum install python-pip python-devel
sudo pip install --upgrade pip
sudo pip install scipy numpy keras
sudo yum install freetype-devel libjpeg-turbo-devel libtiff-devel tcl-devel tk-devel
sudo pip install pillow
sudo pip install h5py # for serializing model weights

# Install tensorflow
# See https://www.tensorflow.org/get_started/os_setup#pip_installation
sudo pip install tensorflow-gpu

# Install cuDNN
# This archive was scp'd to the EC2 instance by provision_gpu.sh
tar -xzvf cudnn*
sudo cp -L cuda/lib64/* $CUDA_HOME/lib64/
sudo cp -L cuda/include/* $CUDA_HOME/include/

# Get my code
sudo yum install git
git clone git@github.com:nhatch/casia

echo "Done"

