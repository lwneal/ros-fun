from keras import backend as K
from keras.models import Model


def gather_layers(model):
    if not hasattr(model, 'layers'):
        return [model]
    layers = []
    for layer in model.layers:
        layers.extend(gather_layers(layer))
    return layers

def gather_states(model):
    states = []
    for layer in gather_layers(model):
        if hasattr(layer, 'states'):
            states.extend(layer.states)
    return states

class Visualizer(object):
    def __init__(self, model):
        self.model = model
        self.int_models = []
        for layer in gather_layers(model):
            self.int_models.append(Model(input=model.input, output=layer.output, name=layer.name))

    def run(self, X):
        print("Activations")
        for model in self.int_models:
            y = model.predict(X)
            print("{: <14}\tstd {: .4f}\tmin {: .4f}\tmax {: .4f}\tabs mean {: .4f}".format(
                y.shape, y.std(), y.min(), y.max(), abs(y).mean()))

    def print_states(self):
        states = gather_states(self.model)
        print("States")
        from shared.util import encode_jpg
        i = 0
        for state in states:
            value = K.get_session().run(state.value())
            print("{: <14} min {} max {}".format(state.name, value.min(), value.max()))
            with open('/tmp/lstm_states_{}.mjpg'.format(i), 'a') as fp:
                fp.write(encode_jpg(value[0].reshape(32,32)))
            i += 1
