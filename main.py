# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from myModel import *
import logger

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    writer = logger.get_logger('C:/Users/chen/Desktop/Myworkspace')
    for i in range(100):
        y = i**2
        writer.add_scalar(tag='test01', scalar_value=y, global_step=i)
    writer.close()
    # a = build_segmentor(dict(type='Net1'))
    # A= register.build_from_cfg(dict(type='Resnet'), MODEL)
    # A=build_from_config(dict(type='net'))


