from keras.models import Model


def gather_layers(model):
    if not hasattr(model, 'layers'):
        return [model]
    layers = []
    for layer in model.layers:
        layers.extend(gather_layers(layer))
    return layers


class Visualizer(object):
    def __init__(self, model):
        self.model = model
        self.int_models = []
        for layer in gather_layers(model):
            self.int_models.append(Model(input=model.input, output=layer.output, name=layer.name))

    def run(self, X):
        for model in self.int_models:
            y = model.predict(X)
            print("{: <18} shape {: <14}\tstd {: .4f}\tmin {: .4f}\tmax {: .4f}\tabs mean {: .4f}".format(
                model.name, y.shape, y.std(), y.min(), y.max(), abs(y).mean()))
