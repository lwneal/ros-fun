import os


video_file = 'video.mjpeg'
timestamp_file = 'timestamps.txt'


def store(frame, timestamp):
    timestamp_str = '{:.03f}\n'.format(timestamp)
    open(video_file, 'ab').write(frame)
    open(timestamp_file, 'a').write(timestamp_str)


def check():
    video_data = open(video_file).read()
    timestamps = open(timestamp_file).readlines()
    return len(video_data), len(timestamps)
