# setup_gpu.sh
#
# This script is intended to be run on a newly created EC2 instance. This instance should be using the free Red Hat AMI and the g2.2xlarge or g2.8xlarge hardware. It will set up the instance to the point that it can run the code in this repo.
#
# I would copy-paste these instructions a few lines at a time. The script has not been tested to be run all at once.


## Copy files from personal computer to EC2 instance.
## These commands should be run on a personal computer.
## HOST_ADDR is the address of the EC2 instance you are setting up.
# cuDNN installer
# Download archive from https://developer.nvidia.com/rdp/cudnn-download. You will need to create an account and answer some questions about intended use.
# Look for the latest version of cuDNN that matches CUDA 7.5 for Linux. Also note that as of this writing, the latest version officially supported by Theano is cuDNN 5.0.
scp -i ~/.ssh/aws_personal.pem ~/Downloads/cudnn-7.5-linux-x64-v* ec2-user@$HOST_ADDR:~
# SSH key for git account
scp -i ~/.ssh/aws_personal.pem ~/.ssh/id_rsa ec2-user@$HOST_ADDR:~/.ssh/id_rsa
# SSH key for other EC2 instances
scp -i ~/.ssh/aws_personal.pem ~/.ssh/aws_personal.pem ec2-user@$HOST_ADDR:~/.ssh/aws_personal.pem
# Training and testing data for CASIA
# See README for information about how to download these. They are large.
# TODO add these files to S3 or somewhere hosted *not* in China
scp -i ~/.ssh/aws_personal.pem HWDB1.1tst_gnt.zip ec2-user@$HOST_ADDR:~
scp -i ~/.ssh/aws_personal.pem HWDB1.1trn_gnt.zip ec2-user@$HOST_ADDR:~


# The remaining commands should be run from the EC2 instance home directory.

sudo yum install gcc-c++ wget vim unzip

# Install CUDA
wget http://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-repo-rhel7-7.5-18.x86_64.rpm
sudo rpm -i cuda-repo-rhel7-7.5-18.x86_64.rpm # ignore NOKEY warning
DKMS_PKG=dkms-2.2.0.3-34.git.9e0394d.el7.noarch.rpm # version might have been updated, you can find the new version by going to rpmfind.net/linux/epel/7/x86_64/d and ctrl-F for 'dkms'
wget ftp://rpmfind.net/linux/epel/7/x86_64/d/${DKMS_PKG}
sudo yum localinstall ${DKMS_PKG} --nogpgcheck
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

echo 'export PATH=$PATH:/usr/local/cuda-7.5/bin' >> .bashrc
echo 'export CUDA_ROOT=/usr/local/cuda-7.5/' >> .bashrc
echo 'export LD_LIBRARY_PATH=${CUDA_ROOT}lib64' >> .bashrc
echo "export THEANO_FLAGS='device=gpu,floatX=float32,force_device=True,lib.cnmem=0.95'" >> .bashrc # for theano (installed later)

# Verify you have an NVIDIA-compatible GPU
sudo yum install pciutils
lspci | grep -i nvidia

# Install NVIDIA driver
sudo yum install kernel-devel
wget http://us.download.nvidia.com/XFree86/Linux-x86_64/361.42/NVIDIA-Linux-x86_64-361.42.run
chmod +x NVIDIA-Linux-x86_64-361.42.run
KERNEL_SOURCE_PATH=`rpm -ql kernel-devel | head -n1`/
sudo ./NVIDIA-Linux-x86_64-361.42.run --kernel-source-path=${KERNEL_SOURCE_PATH}
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

# Install cuDNN
# This archive was scp'd to the EC2 instance in an earlier step
tar -xzvf cudnn*
sudo cp cuda/lib64/* $CUDA_ROOT/lib64/
sudo cp cuda/include/cudnn.h $CUDA_ROOT/include/

# Get my code
sudo yum install git
git clone git@github.com:nhatch/casia
[enter passphrase]
cd casia

# Verify that Theano is configured and able to use the GPU
# check1.py comes from http://deeplearning.net/software/theano/tutorial/using_gpu.html#using-gpu
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python check1.py

