import numpy as np
import random

import capnp
import frametalk_capnp


def point_head_toward_human(preds):
    command = frametalk_capnp.RobotCommand.new_message()
    azumith, altitude = detect_peak(preds)
    command.headRelAzumith = azumith
    command.headRelAltitude = altitude
    command.descriptiveStatement = 'human detected with score {}'.format(preds.max())
    return command


def detect_peak(convnet_map, fov_width_deg=62.0, fov_height_deg=48.6):
    """ Inputs:
    Numpy 2D array (eg. VGG16 layer conv5_1, filter 104)
    Field of View (eg. Kinect 1 FOV is 62x48.6)
    Outputs:
    Horizontal position: Degrees, negative is left, zero is straight ahead
    Vertical position: Degrees, negative is up, zero is straight ahead
    """
    y_pos = np.argmax(np.sum(convnet_map, axis=1))
    x_pos = np.argmax(np.sum(convnet_map, axis=0))

    def to_rel_degrees(position_idx, map_length, fov_degrees):
        return (position_idx * (1.0 / map_length) - 0.5) * fov_degrees

    """
    # Get the weighted average of columns
    idxs = np.arange(0, map_width)
    col_avg = np.average(idxs, weights=np.sum(convnet_map, axis=0))
    """

    map_height, map_width = convnet_map.shape
    azumith = to_rel_degrees(x_pos, map_width, fov_width_deg)
    altitude = to_rel_degrees(y_pos, map_height, fov_height_deg)
    return float(azumith), float(altitude)
