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
        if len(lines) < 2:
            print("Skipping degenerate session {}".format(f))
            continue
        start_time = float(lines[0].split(',')[-1])
        end_time = float(lines[-1].split(',')[-1])
        sessions.append({
            'id': f.rstrip('.timestamps'),
            'duration': end_time - start_time,
            'start_time': float(lines[0].split(',')[-1]),
            'end_time': float(lines[-1].split(',')[-1]),
            'frame_count': len(lines),
        })
    return sessions


def read_frames(session_id):
    # Generator; yields jpg frames read one by one from a mjpeg file
    timestamps_filename = os.path.join(BLOCK_STORAGE_DIR, '{}.timestamps'.format(session_id))
    lines = open(timestamps_filename).read().splitlines()
    video_filename = os.path.join(BLOCK_STORAGE_DIR, '{}.mjpeg'.format(session_id))
    video_fp = open(video_filename)
    for line, next_line in zip(lines, lines[1:]):
        pos, timestamp = parse_timestamp_line(line)
        next_pos, next_timestamp = parse_timestamp_line(next_line)
        frame_data = video_fp.read(next_pos - pos)
        yield timestamp, frame_data
    # last frame
    yield next_timestamp, video_fp.read()


def parse_timestamp_line(line):
    length, timestamp = line.split(',')
    return int(length), float(timestamp)
