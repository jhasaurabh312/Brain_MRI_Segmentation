from keras.applications.vgg16 import VGG16, preprocess_input
vgg16_weight_path="/content/gdrive/My Drive/Brain_mri_clg/weights_model/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
IMG_SIZE = (224,224)
base_model = VGG16(
    weights=vgg16_weight_path,
    include_top=False, 
    input_shape=IMG_SIZE + (3,)
)

from keras.optimizers import Adam, RMSprop
from keras import layers
from keras.models import Model, Sequential

NUM_CLASSES = 1

model = Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(NUM_CLASSES, activation='sigmoid'))

model.layers[0].trainable = False

model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(lr=1e-4),
    metrics=['accuracy']
)

model.summary()