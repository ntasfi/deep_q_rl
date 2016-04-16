#!/bin/bash

echo "==>dependencies setup for deep_q_rl"

echo "==>updating current package..."
sudo apt-get update

echo "==>installing OpenCV..."
sudo apt-get install python-opencv

echo "==>installing Matplotlib..."
sudo apt-get install python-matplotlib python-tk

echo "==>installing Theano ..."
# some dependencies ...
sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git
pip install --user --upgrade --no-deps git+git://github.com/Theano/Theano.git

echo "==>installing Lasagne ..."
pip install --user --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip

# Packages below this point require downloads. 
mkdir build
cd build

if [ ! -d "./pylearn2" ]
then
echo "==>installing Pylearn2 ..."
# dependencies...
sudo apt-get install libyaml-0-2 python-six
git clone git://github.com/lisa-lab/pylearn2
fi
cd ./pylearn2
python setup.py develop --user
cd ..

echo "==>installing Pillow ..."
pip install --user Pillow

if [ ! -d "./pygame" ]
then
echo "==>installing PyGame ..."

#install dependencies for pygame
sudo apt-get install mercurial libav-tools \
    libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsmpeg-dev \
    libsdl1.2-dev  libportmidi-dev libswscale-dev libavformat-dev libavcodec-dev
 
# pygame source
hg clone https://bitbucket.org/pygame/pygame
fi
 
# build and install pygame
cd pygame
python setup.py build
sudo python setup.py install
cd ..

if [ ! -d "./PyGame-Learning-Environment" ]
then
echo "==>installing PLE"

git clone https://github.com/ntasfi/PyGame-Learning-Environment
cd PyGame-Learning-Environment
sudo pip install -e .
cd ..
fi


echo "==>All done!"
