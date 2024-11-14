class Singleton(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "instance"):
            # Create the one and only instance of this class
            cls.instance = super(Singleton, cls).__new__(cls)
        return cls.instance
