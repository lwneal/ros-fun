from keras.models import Model

class Visualizer(object):
    def __init__(self, model):
        self.model = model
        self.int_models = []
        for layer in model.layers:
            self.int_models.append(Model(input=model.input, output=layer.output, name=layer.name))

    def run(self, X):
        for model in self.int_models:
            y = model.predict(X)
            print("{: <20} shape {: <16}\tmean {: .4f}\tstd {: .4f}\tmin {: .4f}\tmax {: .4f}\tabs mean {: .4f}".format(
                model.name, y.shape, y.mean(), y.std(), y.min(), y.max(), abs(y).mean()))
