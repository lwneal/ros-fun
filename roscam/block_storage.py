import time
import os

BLOCK_STORAGE_DIR = '/home/nealla/block_storage'

class BlockStorageContext(object):
    def __init__(self):
        base_filename = os.path.join(BLOCK_STORAGE_DIR, '{}'.format(int(time.time() * 1000)))
        video_filename = '{}.mjpeg'.format(base_filename)
        timestamp_filename = '{}.timestamps'.format(base_filename)
        self.video_fp = open(video_filename, 'ab')
        self.timestamp_fp = open(timestamp_filename, 'a')
        self.video_bytes_written = 0
        print("Started block storage session with filename {}".format(video_filename))

    def store(self, frame, timestamp):
        timestamp_line = '{},{:.03f}\n'.format(self.video_bytes_written, timestamp)
        self.timestamp_fp.write(timestamp_line)
        self.video_bytes_written += len(frame)
        self.video_fp.write(frame)

    def close(self):
        self.video_fp.close()
        self.timestamp_fp.close()


def get_sessions():
    sessions = []
    files = [f for f in os.listdir(BLOCK_STORAGE_DIR) if f.endswith('timestamps')]
    for f in files:
        filename = os.path.join(BLOCK_STORAGE_DIR, f)
        lines = open(filename).readlines()
        if not lines:
            continue
        sessions.append({
            'start_time': float(lines[0].split(',')[-1]),
            'end_time': float(lines[-1].split(',')[-1]),
            'frames': len(lines),
        })
    return sessions
