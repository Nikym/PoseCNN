# PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes

Created by Yu Xiang at [RSE-Lab](http://rse-lab.cs.washington.edu/) at University of Washington and NVIDIA Research.

### Introduction

We introduce PoseCNN, a new Convolutional Neural Network for 6D object pose estimation. PoseCNN estimates the 3D translation of an object by localizing its center in the image and predicting its distance from the camera. The 3D rotation of the object is estimated by regressing to a quaternion representation. [arXiv](https://arxiv.org/abs/1711.00199), [Project](https://rse-lab.cs.washington.edu/projects/posecnn/)

[![PoseCNN](http://yuxng.github.io/PoseCNN.png)](https://youtu.be/ih0cCTxO96Y)

### License

PoseCNN is released under the MIT License (refer to the LICENSE file for details).

### Citation

If you find PoseCNN useful in your research, please consider citing:

    @inproceedings{xiang2018posecnn,
        Author = {Xiang, Yu and Schmidt, Tanner and Narayanan, Venkatraman and Fox, Dieter},
        Title = {PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes},
        Journal   = {Robotics: Science and Systems (RSS)},
        Year = {2018}
    }

# Environment Setup
The following are instructions and information about how to set up the environment in order to experiment with PoseCNN. *Please read all of the instructions before carrying out any work*. This tutorial has been tested on a system with a fresh Ubuntu 16.04 installation.

## Requirements
* Ubuntu 16.04
* An *NVIDIA Graphics Card*
* NVIDIA Drivers >= 410.48 and <= 460.32.3
* CUDA 10.0 (10.0.130)
* CuDNN 7.3.1 (7.3.1.20-1+cuda10.0)
* NCCL 2.4.8
* Bazel 0.10.0
* TensorFlow 1.8 (built from source)

It is important that the CUDA and CuDNN versions are as stated, as TensorFlow 1.8 only supports up to those versions. The NVIDIA driver version can be newer than 410, however.

Graphics card used on the tested machine is a /GeForce GTX 1060 6GB/.

## 0. Required Packages Setup
We need to install a set of packages that are required for some of the later steps. To do so, execute the following set of commands:

```bash
$ pip2 install opencv-python==4.2.0.32
$ sudo apt-get install libopencv-dev
$ pip install mock enum34
$ pip install matplotlib numpy keras Cython Pillow easydict transforms3d
$ sudo apt-get install libsuitesparse-dev libopenexr-dev metis libmetis-dev google-perftools
```

## 1. Setting up NVIDIA software
### 1.1 Graphics Drivers
First we need to install the correct drivers in order to utilise the NVIDIA Graphics card correctly. First we need to add the graphics driver repository to apt / apt-get, so that `apt-get` can retrieve the driver.

```bash
$ sudo add-apt-repository ppa:graphics-drivers/ppa
$ sudo apt update
```

We can then install the required NVIDIA drivers through this. The version mentioned here is the latest at the time of writing (2020/22/12). After the installation is complete we need to reboot.

```bash
$ sudo apt-get install nvidia-460
$ sudo reboot
```

To check if the NVIDIA driver has been installed correctly, use the following command:

```bash
$ nvidia-smi
```

At this stage the output of this command shout indicate the driver version (the one installed), what graphics card is present / detected, and a CUDA version of `N/A`. If the command does not give the expected output, then the installation has not gone right.

Note: If you receive the error message `Failed to initialize NVML: Driver/library version mismatch`, then it is worth trying to reboot again and entering the command once more.

### 1.2 Installing CUDA
This part is a little more fiddly than the previous. For this, we will need to download files through the network using `wget`. To start, navigate to a directory you would like to put the relevant CUDA download files (note, it doesn’t matter where these go as they are just used for installing the actual package and are not the package itself).

First we download the relevant `.deb` file and the `.pub` key to allow us to install the package:

```bash
$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.0.130-1_amd64.deb

$ sudo dpkg -i cuda-repo-ubuntu1604_10.0.130-1_amd64.deb

$ sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
```

Following on from this, we can install the required CUDA version, and reboot.

```bash
$ sudo apt update
$ sudo apt install cuda-10-0
$ sudo reboot
```

When you enter the `nvidia-smi` command again after rebooting, it should indicate a version of CUDA. *Note*, it may show a CUDA version that is *higher* than 10.0, but that is not the case. The version displayed here is the compatible latest version of CUDA for the graphics driver.

To check the actual version of CUDA installed, enter the following command. The output should look something similar to what is shown below.

```bash
$ cat /usr/local/cuda/version.txt
CUDA Version 10.0.130
```

Finally, we need to add export PATH, CUDA_HOME and LD_LIBRARY_PATH variables in our `.bashrc`, to do so enter `nano ~/.bashrc` into the command line and enter the following snippet somewhere:

```
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/extras/CUPTI/lib64"
export CUDA_HOME=/usr/local/cuda-10.0
export PATH="$PATH:/usr/local/cuda-10.0/bin"
```

After doing so, enter the following commands and you should receive a similar output:

```bash
$ source ~/.bashrc
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Sat_Aug_25_21:08:01_CDT_2018
Cuda compilation tools, release 10.0, V10.0.130
```

CUDA and CUDA Toolkit are now installed.

### 1.3 Installing CuDNN
To install CuDNN, we just have to enter the following `apt-get` command into the terminal.

```bash
$ wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
$ sudo apt install ./nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
$ sudo apt-get update
$ sudo apt-get install libcudnn7=7.3.1.20-1+cuda10.0 libcudnn7-dev=7.3.1.20-1+cuda10.0
```

To check the correct version are installed, enter the command below and you should see a similar output.

```bash
$ dpkg -l | grep cudnn
ii  libcudnn7                                  7.3.1.20-1+cuda10.0                             amd64        cuDNN runtime libraries
ii  libcudnn7-dev                              7.3.1.20-1+cuda10.0                             amd64        cuDNN development libraries and headers
```

CuDNN 7.3.1 is now installed.

### 1.4 Installing NCCL
To install NCCL, we need to carry out similar steps to how we installed CUDA. Below are the set of commands we need to enter:

```bash
$ sudo apt install libnccl2=2.4.8-1+cuda10.0 libnccl-dev=2.4.8-1+cuda10.0
```

Following the installation, we should be able to see files `/usr/include/nccl.h` and `/usr/lib/x86_64-linux-gnu/libnccl.so.2`. If they are present, then great! We have installed NCCL successfully.

Because of the way that TensorFlow 1.8 expects NCCL files to be organised, we need to create some symbolic links to those mentioned directories so that those files can be reached later on. To do so, we need to enter the following commands:

```bash
$ sudo mkdir /usr/local/cuda-10.0/nccl
$ sudo ln -s /usr/lib/x86_64-linux-gnu /usr/local/cuda-10.0/nccl/lib
$ sudo ln -s /usr/include /usr/local/cuda-10.0/nccl/include
```

To briefly explain, we can now reach the previously mentioned files using the paths `/usr/local/cuda-10.0/nccl/include/nccl.h` and `/usr/local/cuda-10.0/nccl/lib/libnccl.so.2`, which are the paths that TensorFlow will be looking for.

We also need to copy a license file into this newly created directory. To do so, we enter the following command:

```bash
$ sudo cp /usr/share/doc/libnccl2/copyright /usr/local/cuda-10.0/nccl/NCCL-SLA.txt
```

Check that `NCCL-SLA.txt` has been created successfully. Once it is, we have finished installing and setting up NCCL.

## 2. Setting up TensorFlow
In order to ensure that TensorFlow is built to work around our machine, rather than a general machine, we need to build TensorFlow from source. This section is one of the tricker ones, and may take a long time to carry out some of the steps (such as compiling).

### 2.1 GCC and G++ Versions
Before doing so, we need to ensure that our gcc and g++ versions are set to `4.8.5`. To set up gcc, enter the following set of commands:

```bash
$ sudo apt-get install gcc-4.8
$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 10
$ sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 30
$ sudo update-alternatives --config gcc
```

For the last command above, select option `1` which should correspond to gcc version `4.8.5`.

Then in a similar fashion we can install the correct version of g++.

```bash
$ sudo apt-get install g++-4.8
$ sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 10
$ sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-5 30
$ sudo update-alternatives --config g++
```

Again, select option `1` when prompted, which should correspond to g++ version `4.8.5`.

To check that the correct versions of gcc and g++ are now set, enter the following commands (which should give similar outputs to below):

```bash
$ gcc --version
gcc (Ubuntu 4.8.5-4ubuntu2) 4.8.5
Copyright (C) 2015 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

$ g++ --version
g++ (Ubuntu 4.8.5-4ubuntu2) 4.8.5
Copyright (C) 2015 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

GCC and G++ are now set up.

### 2.2 Installing Bazel
In order to build the TensorFlow package we need to install Bazel version `0.10.0`. To do so here we use a `.sh` file downloaded from the official Bazel GitHub as a means to set up Bazel.

```bash
$ wget https://github.com/bazelbuild/bazel/releases/download/0.10.0/bazel-0.10.0-installer-linux-x86_64.sh
$ chmod +x bazel-0.10.0-installer-linux-x86_64.sh
$ ./bazel-0.10.0-installer-linux-x86_64.sh
```

Once the installation has been completed, there should now be a `bazel@` file inside the `$HOME/bin` directory. To finish our Bazel setup, we add this directory to our `PATH`.

```bash
$ nano ~/.bashrc
Then add:
export PATH="$PATH:$HOME/bin"
```

After executing `source`, when can check the version of Bazel we installed is correct and working. Ensure that the version is indeed `0.10.0`.

```bash
$ source ~/.bashrc
$ bazel version
Build label: 0.10.0
...
```

Bazel is now set up.

### 2.3 Building TensorFlow
This section is by the far longest and more computationally intensive part of the setup, and will take a while for it to be complete. Here we need to get the official TensorFlow repository from GitHub and build release 1.8 from source.

First, we need to clone the repository and go into the correct branch (r1.8).

```bash
$ git clone https://github.com/tensorflow/tensorflow.git
$ cd tensorflow
$ git checkout r1.8
```

Once in the correct branch, we can enter `./configure` into the terminal which will prompt for us to enter a few values and choose some options. Ensure that the python location and package-site is correct. For majority of the options we can enter N, however ensure the following:
	* When prompted if using CUDA, enter `y`.
	* When asked for the CUDA version, enter `10.0`.
	* When asked for the directory of CUDA, enter `/usr/local/cuda-10.0`.
	* When asked for the CuDNN version, enter `7.3.1`.
	* When asked for the CuDNN location, make sure its the same as the directory of CUDA (`/usr/local/cuda-10.0`).
	* When asked for the NCCL version, enter `2.4.8`.
	* When asked for the NCCL location, enter `/usr/local/cuda-10.0/nccl`.

After the configuration is complete, we can start using Bazel to build TensorFlow 1.8. *Note, this process is very time consuming compared to the rest of the steps, and requires processing power*. To begin the build, enter the following command:

```bash
$ bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
```

This will begin building TensorFlow, and will utilise all available cores and as much memory as the build requires. On the test machine this command by itself had caused for the memory and swap usage to be maxed out, causing a slowdown and for the OS to eventually kill the process itself. We can get around this by limiting the available memory and cores by adding a few extra params.

```bash
$ bazel build --config=opt --config=cuda --local_resources=4098,2,1.0 --jobs=2 //tensorflow/tools/pip_package:build_pip_package
```

Here the `—local_resources` param is used to limit the memory usage to `4098MB` and the core usage to `2` cores. A `—jobs` param was also added, which limits the max numbers of “actions” that Bazel can do at one time.

*Note:* Limiting the RAM just by itself is not enough, as for some odd reason or another the C++ compilations do not limit themselves to the value you set, so it is important to also limit the number of cores / jobs if your machine requires it.

After the build is successful, we can create the pip package and install it.

```bash
$ . bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/project/tensorflow_pip
$ pip install ~/project/tensorflow_pip/tensorflow-1.8.0-cp27-cp27mu-linux_x86_64.whl
```

TensorFlow 1.8 is now installed.

## 3. C++ Libraries
Having set up TensorFlow and the NVIDIA software, before we can compile and set up PoseCNN we need to install a few C++ libraries which it relies upon.

It is advised to create a directory where all of the downloaded libraries can be kept together, from where we can carry out the installations. During testing, a directory `~/diss` had been created for this purpose.

Before doing the installations, we also need to ensure we have installed the following libraries.

```bash
$ sudo apt-get install build-essential cmake libgtest-dev libpcl-dev libassimp-dev
```

### 3.1 Eigen
We need Eigen version `3.3.0`.

First, we need to install the `zip` and `unzip` packages for Ubuntu, if they are not already installed.

```bash
$ sudo apt install zip unzip
```

Then we can begin setting up Eigen.

```bash
$ wget https://gitlab.com/libeigen/eigen/-/archive/3.3.0/eigen-3.3.0.zip
$ unzip eigen-3.3.0
$ cd eigen-3.3.0
$ mkdir build
$ cd build
$ cmake ..
$ make
$ sudo make install
```

Eigen3 should now be installed.

### 3.2 Nanoflann
We need Nanoflann version at commit `ad7547f4e6beb1cdb3e360912fd2e352ef959465`.

To install, we carry out the following commands.

```bash
$ wget https://github.com/jlblancoc/nanoflann/archive/ad7547f4e6beb1cdb3e360912fd2e352ef959465.zip
$ unzip ad7547f4e6beb1cdb3e360912fd2e352ef959465.zip -d ./
$ mv nanoflann-ad7547f4e6beb1cdb3e360912fd2e352ef959465 nanoflann
$ cd nanoflann
$ mkdir build
$ cd build
$ cmake ..
$ make && make test
$ sudo make install
```

Nanoflann should now be installed.

### 3.3 Pangolin
We need Pangolin version at commit `1ec721d59ff6b799b9c24b8817f3b7ad2c929b83`.

First, we need to download and uncompress the files.

```bash
$ wget https://github.com/stevenlovegrove/Pangolin/archive/1ec721d59ff6b799b9c24b8817f3b7ad2c929b83.zip
$ unzip 1ec721d59ff6b799b9c24b8817f3b7ad2c929b83 -d ./
$ mv Pangolin-1ec721d59ff6b799b9c24b8817f3b7ad2c929b83 Pangolin
$ cd Pangolin
```

Inside the `CMakeLists.txt` file, we need to add the following line to the top of the file:

```
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
```

After adding and saving, we follow a similar process again.

```bash
$ mkdir build
$ cd build
$ cmake ..
$ cmake --build .
```

Note, if you receive errors regarding GLEW, you can install the package with the following command, and try building again.

```bash
$ sudo apt-get install libglew-dev
```

Pangolin is now installed.

### 3.4 Boost
We require Boost version `1.67.0`.

To install, carry out the following commands.

```bash
$ wget https://dl.bintray.com/boostorg/release/1.67.0/source/boost_1_67_0.zip
$ unzip boost_1_67_0.zip -d ./
$ mv boost_1_67_0 boost
$ cd boost
$ ./bootstrap.sh
$ sudo ./b2
$ sudo ./b2 install
```

Boost is now installed.

### 3.5 Sophus
We require Sophus version in commit `ceb6380a1584b300e687feeeea8799353d48859f`.

To install, carry out the following commands.

```bash
$ wget https://github.com/strasdat/Sophus/archive/ceb6380a1584b300e687feeeea8799353d48859f.zip
$ unzip ceb6380a1584b300e687feeeea8799353d48859f.zip -d ./
$ mv Sophus-ceb6380a1584b300e687feeeea8799353d48859f Sophus
$ cd Sophus
$ mkdir build
$ cd build
$ cmake ..
$ make
$ sudo make install
```

Sophus should now be installed.

### 3.6 NLOPT
Lastly, we need NLOPT version from commit `74e647b667f7c4500cdb4f37653e59c29deb9ee2`.

To install, carry out the following commands.

```bash
$ wget https://github.com/stevengj/nlopt/archive/74e647b667f7c4500cdb4f37653e59c29deb9ee2.zip
$ unzip 74e647b667f7c4500cdb4f37653e59c29deb9ee2.zip -d ./
$ mv nlopt-74e647b667f7c4500cdb4f37653e59c29deb9ee2 nlopt
$ cd nlopt
$ mkdir build
$ cd build
$ cmake ..
$ make
$ sudo make install
```

NLOPT and the rest of the dependencies should now be installed.

## 4.  Setting up PoseCNN
This is the home stretch! Lastly, we will set up PoseCNN. First we need to download the PoseCNN GitHub repository.

```bash
$ git clone https://github.com/yuxng/PoseCNN.git
```

### 4.1 Compiling Kinect Fusion
Before we do this, we need to make a quick slight change to one of our CUDA and Eigen3 files. To do so, execute the following commands.

```bash
$ sudo chmod 777 /usr/local/cuda-10.0/include/crt/common_functions.h
$ sudo nano /usr/local/cuda-10.0/include/crt/common_functions.h
```

Then on line 74, comment out the statement `#define __CUDACC_VER__ "__CUDACC_VER__ is no longer supported. Use __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, and __CUDACC_VER_BUILD__ instead."`.

Save the file, then execute the following commands

```bash
$ sudo chmod 774 /usr/local/include/eigen3/Eigen/Core
$ sudo nano /usr/local/include/eigen3/Eigen/Core
```

You should then locate the line that states `#include <math_functions.hpp>` and modify it to:

```
#include <math_functions.h>
```

This is because the CUDA version we are using had migrated from using the `.hpp` file extension to just a `.h` one.

After that is complete, we can execute the following commands.

```bash
$ cd PoseCNN
$ cd lib/kinect_fusion
$ mkdir build
$ cd build
$ cmake ..
$ make
```

Kinect Fusion should now be compiled. After this, go back to the `common_functions.h` file and uncomment the line we previously commented out.

### 4.2 Compiling Synthesize
Return back into the `lib` directory, and enter the following set of commands.

```bash
$ cd synthesize
$ mkdir build
$ cd build
$ cmake ..
$ make
```

### 4.3 Compile Layers
Before compiling the layers using `make.sh`, we need to change a value inside certain files under the `lib` directory. To see which files need to be modified, enter the following command when inside the `$ROOT/lib` directory.

```bash
$ grep -rlw "kThreadsPerBlock = 1024" .
```

This outputs a list of files containing the given string. Go inside those files and modify `kThreadsPerBlock` to be 512 instead of 1024, and save. After that is done, we can compile the layers by running `make.sh`. Note, if you have cloned the forked repository (`Nikym/PoseCNN`), then the modification of `kThreadsPerBlock` has already been done for you.

After doing so, we need to run the python setup script that is in the same directory. Note, for this to be successful we must follow the steps in section 2.1 and briefly change the GCC and G++ versions to the their original (5+). After doing so, run the following command.

```bash
$ python setup.py build_ext --inplace
```

Once the script has finished executing, add the following export to your bash profile.

```
export PYTHONPATH=$PYTHONPATH:$ROOT/lib:$ROOT/lib/synthesize/build
```

### 4.4 Download VGG16 Weights
Weights for the initial part of the CNN (VGG16) are already adjusted and available for us to download. To do so, enter the following command inside `$ROOT/data/imagenet_models`, where `$ROOT` is the root directory of your cloned PoseCNN.

```bash
$ wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1UdmOKrr9t4IetMubX-y-Pcn7AVaWJ2bL' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1UdmOKrr9t4IetMubX-y-Pcn7AVaWJ2bL" -O "vgg16.npy" && rm -rf /tmp/cookies.txt
```

### 4.5 Download the YCB Models
In order to visualise the results and know the default rotations etc of the objects, we need to download the relevant models. First, navigate to `$ROOT/data/data_models` then run the following command to download the zipped YCB models used by PoseCNN.

```bash
$ wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1UNJ56Za6--bHGgD3lbteZtXLC2E-liWz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1UNJ56Za6--bHGgD3lbteZtXLC2E-liWz" -O "demo_models.zip" && rm -rf /tmp/cookies.txt
```

After the download is finished, unzip it.

```bash
$ unzip demo_models.zip
```

### 4.6 Download External Datasets
In order to run the network you must have two datasets already downloaded. Download the following two datasets.

```bash
$ wget https://groups.csail.mit.edu/vision/SUN/releases/SUN2012.tar.gz
$ wget ftp://cs.stanford.edu/cs/cvgl/ObjectNet3D/ObjectNet3D_images.zip
```

Unzip the contents and place them into `$ROOT/data/SUN2012/data` and `$ROOT/data/ObjectNet3D/data` directories respectively.

### 4.7 Download Training Dataset
In this step, the dataset that is intended to be used for training should be downloaded as well as the object models.

The original dataset and models can be found by following [this link](https://rse-lab.cs.washington.edu/projects/posecnn/). After the required datasets and models have been downloaded and unzipped, navigate to `$ROOT/data/LOV` and enter the following commands.

```bash
$ ln -s PATH/TO/DATASET data
$ ln -s PATH/TO/MODELS models
```

It is also possible to directly place the dataset into a real directory `data` inside the `$ROOT/data/LOV` directory, however it is recommended that symbolic links are used.

## 5. Running PoseCNN
You should now be able to run the network. To start using the network, here are a list of commands to use. Note, all are ran from `$ROOT`.

```bash
# Train the network
$ ./experiments/scripts/fat_data_train.sh 0

# Test the network
$ ./experiments/scripts/fat_data_test.sh 0
```

Note, the `0` refers to the GPU device ID, where the first GPU will always be labeled as 0. If faults persist, ensure that the NVIDIA drivers are set up correctly by running the `nvidia-smi` command.