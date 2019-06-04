import logging
import multiprocessing as mp
import ctypes
import numpy as np
import cv2
import random

logger = logging.getLogger(__name__)

PROCESSES = 16
MAX_FLOAT_NUM = 512 * 3 * 32 * 224 * 224
ret_base = mp.RawArray(ctypes.c_float, MAX_FLOAT_NUM)
counter = mp.RawValue(ctypes.c_int, 0)


def sample_clip_func((filename, p, batch_size, n_frame, crop_size, scale_w, scale_h, is_train, temporal_center)):
    ret = np.frombuffer(ret_base, dtype=np.float32, count=batch_size * 3 * n_frame * crop_size * crop_size).reshape(
        (batch_size, 3, n_frame, crop_size, crop_size))

    tmp = None
    invalid_frame = 0
    while tmp is None:
        v = cv2.VideoCapture(filename)
        width, height, length = v.get(cv2.CAP_PROP_FRAME_WIDTH), v.get(cv2.CAP_PROP_FRAME_HEIGHT), \
                                v.get(cv2.CAP_PROP_FRAME_COUNT) - invalid_frame
        assert crop_size <= width and crop_size <= height, \
            '{0} <= {1} ; {2} <= {3} ; {4} <= {5} ; filename'.format(n_frame, length, crop_size, width, crop_size, height, filename)
            #'%d <= %d ; %d <= %d ; %d <= %d ; filename' % (n_frame, length, crop_size, width, crop_size, height, filename)
        length = int(length)
        if length < n_frame:
            logger.info("{0} length {1} < {2}".format(filename, length, n_frame))

        if not is_train and temporal_center:
            frame_st = 0 if length <= n_frame else int((length - n_frame) / 2)
        else:
            frame_st = 0 if length <= n_frame else random.randrange(length - n_frame + 1)

        if is_train:
            row_st = random.randrange(scale_h - crop_size + 1)
            col_st = random.randrange(scale_w - crop_size + 1)
        else:
            row_st = int((scale_h - crop_size) / 2)
            col_st = int((scale_w - crop_size) / 2)

        tmp = np.zeros((n_frame, crop_size, crop_size, 3), dtype=np.float32)
        v.set(cv2.CAP_PROP_POS_FRAMES, frame_st)

        for frame_p in xrange(min(n_frame, length)):
            _, f = v.read()
            if f is not None:
                f = cv2.resize(f, (scale_w, scale_h))
                tmp[frame_p, ...] = f[row_st:row_st + crop_size, col_st:col_st + crop_size, :]
            else:
                tmp = None
		invalid_frame += 1
                counter.value += 1
		print ("Invalid Filename", filename, " invalid_frame:", invalid_frame, " length:", length)
		logger.debug("Counter: {0}".format(counter.value))
                break

    # tmp is D,H,W,C
    # Temporal transform: looping
    if length < n_frame:
        tmp[-(n_frame - length):] = tmp[:(n_frame - length)]

    tmp = tmp.transpose((3, 0, 1, 2))
    # now tmp is C,D,H,W

    # random flip the video horizontally
    if is_train and random.choice([True, False]):
        tmp = np.flip(tmp, 3)

    ret[p, ...] = tmp


def sample_clips(filenames, batch_size, n_frame, crop_size, scale_w=171, scale_h=128, is_train=True,
                 temporal_center=False):
    ret = np.frombuffer(ret_base, dtype=np.float32, count=batch_size * 3 * n_frame * crop_size * crop_size).reshape(
        (batch_size, 3, n_frame, crop_size, crop_size))
    #print ("get clip")
    process_pool.map(sample_clip_func, [(filenames[p], p, batch_size, n_frame, crop_size, scale_w, scale_h, is_train,
                                         temporal_center) for p in xrange(len(filenames))])

    #for p in xrange(len(filenames)):
    #	sample_clip_func((filenames[p], p, batch_size, n_frame, crop_size, scale_w, scale_h, is_train))
    if counter.value and counter.value % 10 == 0:
        logger.info("Invalid counter {0}".format(counter.value))
        counter.value += 1
    assert ret.dtype == np.float32 and ret.shape == (batch_size, 3, n_frame, crop_size, crop_size)
    #print("clip done")
    # normalize here
    m = np.mean(ret, axis=(0, 2, 3, 4))
    std = np.std(ret, axis=(0, 2, 3, 4))
    for i in range(3):
        ret[:, i, :, :, :] = (ret[:, i, :, :, :] - m[i]) / (std[i] + 1e-3)

    return ret


process_pool = mp.Pool(processes=PROCESSES)
