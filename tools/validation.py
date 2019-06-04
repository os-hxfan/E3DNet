import argparse
import os
import numpy as np
import mxnet as mx

from models.E3DNet import create_m3d
from lib.data import ClipBatchIter
import time


def validation(args):
    gpus = [int(i) for i in args.gpus.split(',')]
    if len(gpus) == 0:
        kv = None
    else:
        kv = mx.kvstore.create('local')

    total_batch_size = args.batch_per_device * len(gpus)
    # Load pretrained models
    sym, arg_params, aux_params = mx.model.load_checkpoint(os.path.join(args.output, args.model_prefix), args.eval_epoch)
     
    m = mx.module.Module(sym, context=[mx.gpu(i) for i in gpus]) #, fixed_param_names=fixed_params)

    # Load Validation data
    val_data_iter = ClipBatchIter(datadir=args.datadir, batch_size=total_batch_size, n_frame=args.n_frame,
                                  crop_size=args.crop_size, train=False, scale_w=args.scale_w, scale_h=args.scale_h,
                                  temporal_center = True)
    val_data = mx.io.PrefetchingIter(val_data_iter)

    m.bind(data_shapes=val_data.provide_data, label_shapes=val_data.provide_label, for_training=False)
    
    m.set_params(arg_params, aux_params, allow_missing=True)


    n_label = len(val_data_iter.clip_lst)
    n_batch = n_label // total_batch_size + 1 if n_label % total_batch_size else 0
    outputs = np.zeros((n_batch, total_batch_size, args.num_class))
    labels = np.array(val_data_iter.clip_lst)[:, 1].astype(int)

    for i in range(args.clips_per_video):
        val_data.reset()
        data_iter = iter(val_data)
        end_of_batch = False
        next_data_batch = next(data_iter)

        start_time = time.time()
        i_batch = 0
        while not end_of_batch:
            data_batch = next_data_batch
            m.forward(data_batch, is_train=False)
            batch_outputs = m.get_outputs()[0].asnumpy()
            outputs[i_batch] += batch_outputs
            i_batch += 1
            try:
                # pre fetch next batch
                next_data_batch = next(data_iter)
                m.prepare(next_data_batch)
            except StopIteration:
                end_of_batch = True
        elapsed_time = time.time() - start_time
        tmp_outputs = np.reshape(outputs, (n_batch * total_batch_size, args.num_class))[: n_label]
        tmp_outputs = np.argmax(tmp_outputs, axis=1)
        diff = (tmp_outputs == labels)
        acc = sum(diff) * 1.0 / len(diff)
        print("epoch %d, used time %.3f, %.3f per batch, %.3f per clip, acc %.4f" % (i, elapsed_time,
                elapsed_time / n_batch, elapsed_time / n_label, acc))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="command for training r2plus1d network")
    parser.add_argument('--gpus', type=str, default='0', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--output', type=str, default='./models/', help='the output directory')
    parser.add_argument('--model_prefix', type=str, default="", help='model prefix')
    parser.add_argument('--eval_epoch', type=int, default=1, help='the epoch num to evaluate')
    parser.add_argument('--datadir', type=str, default='/mnt/truenas/scratch/yijiewang/deep-video/deep-p3d/UCF101/',
                        help='the UCF101 datasets directory')
    parser.add_argument('--batch_per_device', type=int, default=4, help='the batch size')
    parser.add_argument('--clips_per_video', type=int, default=1, help='the number of epoch')
    parser.add_argument('--n_frame', type=int, default=32, help='the number of frame to sample from a video')
    parser.add_argument('--crop_size', type=int, default=112, help='the size of the sampled frame')
    parser.add_argument('--scale_w', type=int, default=171, help='the rescaled width of image')
    parser.add_argument('--scale_h', type=int, default=128, help='the rescaled height of image')
    parser.add_argument('--num_class', type=int, default=101, help='the number of class')

    args = parser.parse_args()
    validation(args)


