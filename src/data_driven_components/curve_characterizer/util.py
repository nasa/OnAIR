import numpy as np

classes = {0:'LINEAR INCREASE',
           1:'LINEAR DECREASE',
           2:'SINUSOIDAL',
           3:'FLAT'}

def get_offset(mode, frame_size):
    if mode == 'midpoint':
        return int(frame_size/2)
    if mode == 'lookback':
        return frame_size
    if mode == 'lookahead':
        return 0

def midpoint(A, model, frame_size):
    results = []
    
    half_fs = int(frame_size/2)
    i = half_fs
    while i < len(A)-half_fs:
        B = np.array(A[i-half_fs:i+half_fs]).reshape(1,frame_size,1)
        results.append(model.predict([B,B,B]))
        i = i +1

    characterization = []
    for res in results:
        characterization.append(classes[np.argmax(res)])

    color_spans = []
    for i in range(len(characterization)):
        start = i+half_fs
        end = i+half_fs+1
        color_spans.append((start, end))

    return characterization, color_spans

def lookback(A, model, frame_size):
    results = []
    i = frame_size
    while i < len(A):
        B = np.array(A[i-frame_size:i]).reshape(1,frame_size,1)
        results.append(model.predict([B,B,B]))
        i = i +1

    characterization = []
    for res in results:
        characterization.append(classes[np.argmax(res)])

    color_spans = []
    for i in range(len(characterization)):
        start = i+frame_size
        end = i+frame_size-1
        color_spans.append((start, end))

    return characterization, color_spans


def lookahead(A, model, frame_size):
    results = []
    i = 0
    while i < len(A)-frame_size:
        B = np.array(A[i:i+frame_size]).reshape(1,frame_size,1)
        results.append(model.predict([B,B,B]))
        i = i +1

    characterization = []
    for res in results:
        characterization.append(classes[np.argmax(res)])


    color_spans = []
    for i in range(len(characterization)):
        start = i
        end = i+1
        color_spans.append((start, end))

    return characterization, color_spans

def chunk_out_characterization(char_list, frame_size, mode='midpt'):
    current_char = char_list[0]
    indices = []
    start = 0
    for i in range(len(char_list)):
        if char_list[i] != current_char:
          hfs = int(frame_size/2)
          indices.append((start+hfs, i+hfs))
          current_char = char_list[i]
          start = i + 1

    indices = [x for x in indices if (x[1]-x[0] > int(frame_size/2))]

    return indices


def recolor_points(orig_chars, new_chars, big_frame_size, small_frame_size, chunkies, mode='lookback'):
    offset = get_offset(mode, big_frame_size)
    prune_size = int((len(new_chars) - len(orig_chars))/2)

    new_chars = new_chars[prune_size:len(new_chars) - prune_size]
    for i in range(len(chunkies)):
        chunk_end = chunkies[i][1]
        offset_chunk_end = chunk_end - offset
        for j in range(small_frame_size):
            orig_chars[offset_chunk_end] = new_chars[offset_chunk_end]
    return orig_chars







