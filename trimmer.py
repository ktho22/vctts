import webrtcvad, os, wave, contextlib, collections, argparse

def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    output = []
    while offset + n < len(audio):
        output.append(Frame(audio[offset:offset + n], timestamp, duration))
        timestamp += duration
        offset += n
    return output

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, frames, filename):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    aggressiveness = 3
    while aggressiveness >= 0:
        vad = webrtcvad.Vad(aggressiveness)
        voiced_frames = []
        for frame in frames:
            if not triggered:  #unvoiced part
                ring_buffer.append(frame)
                num_voiced = len([f for f in ring_buffer
                                  if vad.is_speech(f.bytes, sample_rate)])
                if num_voiced > 0.3 * ring_buffer.maxlen:
                    triggered = True
                    voiced_frames.extend(ring_buffer)
                    ring_buffer.clear()
            else:  #voiced part
                voiced_frames.append(frame)
                ring_buffer.append(frame)
                num_unvoiced = len([f for f in ring_buffer
                                    if not vad.is_speech(f.bytes, sample_rate)])
                if num_unvoiced > 0.3 * ring_buffer.maxlen:
                    triggered = False
                    ring_buffer.clear()
        if voiced_frames:
            return b''.join([f.bytes for f in voiced_frames])
        aggressiveness -= 1
    print('Could not find voice activity at', filename)
    return b''.join([f.bytes for f in frames])

def trim(rDirectory, wDirectory):
    frame_duration_ms = 30
    padding_duration_ms = 300
    # print("read dir: ", rDirectory)
    for root, dirnames, filenames in os.walk(rDirectory):
        for filename in filenames:
            if filename[-4:] == '.wav':
                rf = os.path.join(root, filename)
                audio, sample_rate = read_wave(rf)
                frames = frame_generator(frame_duration_ms, audio, sample_rate)
                segment = vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, frames, rf)
                wPath = str(wDirectory + '/' + filename)
                write_wave(wPath, segment, sample_rate)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='training script')
    parser.add_argument('--in_dir', type=str, help='type dataset for trimming')
    parser.add_argument('--out_dir', type=str, help='type dataset for trimming')
    args = parser.parse_args()

    if not args.in_dir or not args.out_dir:
        parser.error('--in_dir and --out_dir should be given')
        
    in_dir = args.in_dir
    out_dir = args.out_dir

    # ------ trimming scilence using VAD
    os.makedirs(out_dir, exist_ok=True)
    trim(in_dir, out_dir)
