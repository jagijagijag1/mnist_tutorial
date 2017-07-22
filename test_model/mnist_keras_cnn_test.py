import sys
import numpy as np
np.random.seed(1337)

from keras.models import model_from_json
from PIL import Image


# set paramters
json_model = 'mnist_keras_cnn_model.json'
weights = 'mnist_keras_cnn_weights.hdf5'
target_img = "img/7.png"

# import a model
model = model_from_json(open(json_model).read())
model.load_weights(weights)

# import an image
image = Image.open(target_img).convert('L')
image = image.resize((28, 28), Image.ANTIALIAS)
data = np.asarray(image, dtype=float)
data = data.reshape(1, 28, 28, 1)

# for white background image, inverse brack/white
# if the background of the input image is black, comment out below
for i in range(len(data)):
    xs = data[i]
    for j in range(len(xs)):
        data[i][j] = 255 - data[i][j]

# normalize
data = data.astype('float32')
data /= 255

# display imgae as text
disp = np.asarray(image, dtype=int)
for xs in disp:
    for x in xs:
        if x == 255:
            sys.stdout.write('   ' % x)
        else:
            sys.stdout.write('XXX' % x)
    sys.stdout.write('\n')

# predict
classes = model.predict_classes(data, batch_size=32)

# result
print("predicted as ", classes[0])
