import logging
import argparse
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mxnet as mx
from models.E3DNet import create_m3d
from lib.data import ClipBatchIter
from mxnet import profiler

train_list = ["fc", "comp_17", "comp_16", "comp_15", "comp_14", "softmax"]
tmp_pool_list = ["final_fc", "softmax_label"]
profiler.set_config(profile_all=True, aggregate_stats=True, filename='profile_output_m3d.json')


def plot_schedule(schedule_fn, iterations=1500):
    # Iteration count starting at 1
    iterations = [i+1 for i in range(iterations)]
    lrs = [schedule_fn(i) for i in iterations]
    plt.scatter(iterations, lrs)
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")
    #plt.savefig('learning_rate.png')

def train(args):
    gpus = [int(i) for i in args.gpus.split(',')]
    num_gpus = len(gpus)

    logging.info("number of gpu %d" % num_gpus)

    if len(gpus) == 0:
        kv = None
    else:
        #kv = mx.kvstore.create('device')
        kv = mx.kvstore.create('local')
    logging.info("Running on GPUs: {}".format(gpus))

    # Modify to make it consistent with the distributed trainer
    total_batch_size = args.batch_per_device * num_gpus

    # Round down epoch size to closest multiple of batch size across machines
    epoch_iters = int(args.epoch_size / total_batch_size)
    args.epoch_size = epoch_iters * total_batch_size
    logging.info("Using epoch size: {}".format(args.epoch_size))
    
    if args.pretrained_dir:
        #Load from pretrained model
        print("Loading from pretrained model:", args.pretrained_dir)
        if (args.load_epoch != 0):
            arg_params = {}
            aux_params = {}
            pre_net, arg_params, aux_params = mx.model.load_checkpoint(os.path.join(args.pretrained_dir, args.model_prefix), args.load_epoch)
	    
            net = create_m3d(
		num_class=args.num_class,
		no_bias=True,
		model_depth=args.model_depth,
		final_spatial_kernel=7 if args.crop_size == 112 else 14,
		final_temporal_kernel=int(args.n_frame / 8),
		bn_mom=args.bn_mom,
		cudnn_tune=args.cudnn_tune,
		workspace=args.workspace,
	    )
            
        else:
            sys.exit("Require parameters: model_prefix and load_epoch")

    else:
        print("Creating the network from scratch")
        # Create Network
        arg_params = {}
        aux_params = {}
        net = create_m3d(
            num_class=args.num_class,
            no_bias=True,
            model_depth=args.model_depth,
            final_spatial_kernel=7 if args.crop_size == 112 else 14,
            final_temporal_kernel=int(args.n_frame / 8),
            bn_mom=args.bn_mom,
            cudnn_tune=args.cudnn_tune,
            workspace=args.workspace,
        )
    arg_params_list = list(arg_params.keys())
    aux_params_list = list(aux_params.keys())
    fixed_arg_params = []
    fixed_aux_params = []
    del_arg_params_name = []
    del_aux_params_name = []
    for arg_item in arg_params_list:
        fixed = True
        for tmp_pool_item in tmp_pool_list:
            if (tmp_pool_item in arg_item):
                del_arg_params_name.append(arg_item)
        for train_item in train_list:
            if (train_item in arg_item):
                fixed = False
                break  
        if fixed:
            fixed_arg_params.append(arg_item)
 
    for aux_item in aux_params_list:
        fixed = True
        for tmp_pool_item in tmp_pool_list:
            if (tmp_pool_item in aux_item):
                del_aux_params_name.append(aux_item)
        for train_item in train_list:
            if (train_item in aux_item):
                fixed = False
                break  
        if fixed:
            fixed_aux_params.append(aux_item)
     
    #print ("!!!Before:", len(arg_params.keys()))
    #print ("", )
    for del_arg_item in del_arg_params_name:
        if del_arg_item in arg_params:
            print ("arg_param:", del_arg_item, " Total:", len(arg_params))
            del arg_params[del_arg_item]

    for del_aux_item in del_aux_params_name:
        if del_aux_item in aux_params:
            print ("axu_param:", del_aux_item)
            del aux_params[del_aux_item]
    '''
    print ("Total arg in net:", len(net.list_arguments()))
    print ("Loaded arguement:", len(arg_params.keys()))
    print ("Total aux in net:", len(net.list_auxiliary_states()))
    print ("Loded aux state:", len(aux_params.keys()))
    '''
    #print ("!!!aux_params:", aux_params_list)
    args_symbol = net.list_arguments()
    for arg in args_symbol:
        if arg not in arg_params:
            print("arg %s not loaded" % arg)    

    for arg in arg_params:
        if arg not in args_symbol:
            print("arg %s not used in net" % arg)
    #print ("!!!fixed arg_params:", fixed_arg_params)
    #print ("!!!fixed aux_params:", fixed_aux_params)
    # Create Module
    m = mx.module.Module(net, context=[mx.gpu(i) for i in gpus])#, fixed_param_names=fixed_arg_params) #, fixed_param_names=fixed_params)
    if args.plot:
        v = mx.viz.plot_network(net, title='R2Plus1D-train',
                                shape={'data': (total_batch_size, 3, args.n_frame, args.crop_size, args.crop_size)})
        v.render(filename='models/R2Plus1D-train', cleanup=True)
    print ("Prepare the train data and eval data")
    train_data = mx.io.PrefetchingIter(ClipBatchIter(datadir=args.datadir, batch_size=total_batch_size,
                                                     n_frame=args.n_frame, crop_size=args.crop_size, train=True,
                                                     scale_w=args.scale_w, scale_h=args.scale_h, epoch_iter=args.epoch_size))
    print ("Complete the train data")
    eval_data = mx.io.PrefetchingIter(ClipBatchIter(datadir=args.datadir, batch_size=total_batch_size,
                                                    n_frame=args.n_frame, crop_size=args.crop_size, train=False,
                                                    scale_w=args.scale_w, scale_h=args.scale_h,
                                                    temporal_center=True))
    print ("Complete the val data")
    # Set optimizer
    optimizer = args.optimizer
    optimizer_params = {}
    optimizer_params['learning_rate'] = args.lr
    optimizer_params['momentum'] = args.momentum
    optimizer_params['wd'] = args.wd

    if args.lr_scheduler_step:
        optimizer_params['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(step=int(args.lr_scheduler_step * epoch_iters),
                                                                           factor=args.lr_scheduler_factor)
	plot_schedule(optimizer_params['lr_scheduler'])
    
    #print ("Runing the training")
    #profiler.set_state('run')
    m.fit(
        train_data=train_data,
        eval_data=eval_data,
        eval_metric='accuracy',
        epoch_end_callback=mx.callback.do_checkpoint(args.output + '/test', 1),
        batch_end_callback=mx.callback.Speedometer(total_batch_size, 20),
        kvstore=kv,
        optimizer=optimizer,
        optimizer_params=optimizer_params,
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
        arg_params=arg_params,
        aux_params=aux_params,
        allow_missing=True,
        begin_epoch=args.begin_epoch,
        num_epoch=args.num_epoch,
    )
    #profiler.set_state('stop')
    #print(profiler.dumps())

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="command for training p3d network")
    parser.add_argument('--gpus', type=str, default='0', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--plot', type=int, default=0, help='plot the network architecture')
    parser.add_argument('--pretrained_dir', type=str, default='', help='pretrained model path')
    parser.add_argument('--datadir', type=str, default='/mnt/truenas/scratch/yijiewang/deep-video/deep-p3d/UCF101/',
                        help='the UCF101 datasets directory')
    parser.add_argument('--output', type=str, default='./output/', help='the output directory')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
    parser.add_argument('--cudnn_tune', type=str, default='off', help='optimizer')
    parser.add_argument('--workspace', type=int, default=512, help='workspace for GPU')
    parser.add_argument('--lr_scheduler_step', type=int, default=0, help='reduce lr after n step')
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.1, help='lr scheduler factor')
    parser.add_argument('--lr', type=float, default=1e-4, help='initialization learning rate')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay for sgd')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--bn_mom', type=float, default=0.9, help='momentum for bn')
    parser.add_argument('--batch_per_device', type=int, default=4, help='the batch size')
    parser.add_argument('--batch_size', type=int, default=16, help='the batch size')
    parser.add_argument('--num_class', type=int, default=101, help='the number of class')
    parser.add_argument('--model_depth', type=int, default=34, help='network depth')
    parser.add_argument('--num_epoch', type=int, default=90, help='the number of epoch')
    parser.add_argument('--epoch_size', type=int, default=100000, help='the number of epoch')
    parser.add_argument('--begin_epoch', type=int, default=0, help='begin training from epoch begin_epoch')
    parser.add_argument('--n_frame', type=int, default=32, help='the number of frame to sample from a video')
    parser.add_argument('--crop_size', type=int, default=112, help='the size of the sampled frame')
    parser.add_argument('--scale_w', type=int, default=171, help='the rescaled width of image')
    parser.add_argument('--scale_h', type=int, default=128, help='the rescaled height of image')
    parser.add_argument('--load_epoch', type=int, default=0, help='the epoch of pretrained model')
    parser.add_argument('--model_prefix', type=str, default="", help='model prefix')

    args = parser.parse_args()

    # Create Output Dir
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Set Logger
    logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(args.output, 'log.txt'),
                        filemode='w')
    # Define a new Handler to log to console as well
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)

    # Start training
    logging.info(" ".join(sys.argv))
    logging.info(args)

    train(args)
