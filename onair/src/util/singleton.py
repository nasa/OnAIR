# class Singleton(object):
#     _instance = None
#     def __new__(class_, *args, **kwargs):
#         if not isinstance(class_._instance, class_):
#             class_._instance = object.__new__(class_, *args, **kwargs)
#         return class_._instance
    

class Singleton(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            # Create the one and only instance of this class
            cls.instance = super(Singleton, cls).__new__(cls)
        return cls.instance