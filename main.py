import logger
import torch.multiprocessing as mp
from tqdm import tqdm,trange
from train import myfunc
import torch
if __name__ == '__main__':
    num_GPU = torch.cuda.device_count()
    print('num_GPU : %d' %num_GPU)
    mp.spawn(fn=myfunc, args=(num_GPU), nprocs=num_GPU)

    # writer = logger.get_logger('C:/Users/chen/Desktop/Myworkspace')
    # model = UNet()
    # for i in range(100):
    #     y = i**2
    #     writer.add_scalar(tag='test01', scalar_value=y, global_step=i)
    # writer.close()
    # a = build_segmentor(dict(type='Net1'))
    # A= register.build_from_cfg(dict(type='Resnet'), MODEL)
    # A=build_from_config(dict(type='net'))


