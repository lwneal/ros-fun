import time
import h5py

# Load weights from an HDF5 file into a Keras model
# Requires that each layer in the model have a unique name
def load_weights(model, filename):
    start_time = time.time()

    # Collect nested layers (eg. layers inside a Merge)
    model_matrices = {}
    model_layers = {}
    def collect_layers(item):
        for layer in item.layers:
            if hasattr(layer, 'layers'):
                collect_layers(layer)
            model_layers[layer.name] = layer
            for mat in layer.weights:
                if mat.name in model_matrices and model_matrices[mat.name] is not mat:
                    print("Warning: Found more than one layer named {} in model {}".format(mat.name, model))
                model_matrices[mat.name] = mat
    collect_layers(model)

    print("Loading weights from filename {} to model {}".format(filename, model))
    f = h5py.File(filename)

    group = f['model_weights']
    for layer_name in group:
        saved_layer = group[layer_name]
        layer_mats = [saved_layer[name].value for name in saved_layer]

        # TODO: Saved matrices are in this order:
        saved_mats = {name: saved_layer[name].value for name in saved_layer}
        # But they need to be in this order:
        model_mat_names = [w.name for w in model_layers[layer_name].weights]
        # Put layer_mats into the correct order

        sorted_weight_mats = sort_weight_mats(saved_mats, model_mat_names)
        model_layers[layer_name].set_weights(sorted_weight_mats)

    print("Loaded {} layers in {:.2f}s".format(len(model_layers), time.time() - start_time))
    f.close()


def sort_weight_mats(saved_mats, model_mat_names):
    mats = []
    for name in model_mat_names:
        mats.append(saved_mats[name])
    return mats
