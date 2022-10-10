# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import register

MODEL = register.Registry('models')


@MODEL.register_module()
class net():
    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    A=MODEL.build(dict(type='net'))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
