import sys
import time
import numpy as np
from resnet50 import ResNet50
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions

if len(sys.argv) < 3:
    print("Usage: {} input.jpg output.jpg".format(sys.argv[0]))
    exit()
img_path = sys.argv[1]
output_path = sys.argv[2]

start_time = time.time()
print "loading resnet..."
model = ResNet50(weights='imagenet')
print "finished loading resnet in {} seconds".format(time.time() - start_time)

start_time = time.time()
print "loading image"

WIDTH=224

# 414 backpack
# 423 barber_chair
# 559 folding_chair
# 624 library
# 765 rocking_chair
# 285 Egyptian cat
CLASS = 285

full_image = image.load_img(img_path, target_size=(WIDTH*4, WIDTH*4))

resnet_output = np.zeros((WIDTH*4, WIDTH*4), dtype=float)

start_time = time.time()
STEP = 16
for dx in range(0, WIDTH*4 - WIDTH/2, STEP):
    for dy in range(0, WIDTH*4 - WIDTH/2, STEP):
        # View 25% of the image at a time
        img = full_image.crop((dx, dy, dx + WIDTH, dy + WIDTH))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)

        pos_x = dx + WIDTH/2
        pos_y = dy + WIDTH/2
        for i in range(STEP):
            for j in range(STEP):
                resnet_output[pos_y + i][pos_x + j] = preds[0][CLASS]
        print dx, dy, preds[0][CLASS]

from PIL import Image
resnet_output *= 255.0 / resnet_output.max()
img = Image.fromarray(resnet_output).convert('RGB')
img.save(output_path)
