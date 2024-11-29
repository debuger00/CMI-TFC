""" dynamically load settings

author axiumao
"""
import conf.global_settings as settings

class Settings:
    def __init__(self, settings):

        for attr in dir(settings):  # dir() 返回settings 对象所有的属性，
            if attr.isupper():  # 筛选出名称为大写的属性（通常是全局常量）
                setattr(self, attr, getattr(settings, attr)) #这些大写属性动态添加为 Settings 实例的属性，并赋值为对应的值（通过 getattr(settings, attr) 获取原值）

settings = Settings(settings)
