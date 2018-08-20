#pip3 install pydot
#sudo apt-get instal python3-graphviz

from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.utils import plot_model

model = InceptionV3()
plot_model(model, to_file='Inception v3.png')
model = VGG16()
plot_model(model, to_file='VGG16_.png')
model = VGG19()
plot_model(model, to_file='VGG19.png')
model = ResNet50()
plot_model(model, to_file='ResNet50.png')
