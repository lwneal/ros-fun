
def build_model():
    from keras.models import Sequential
    from keras.layers import Convolution2D
    
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1)
