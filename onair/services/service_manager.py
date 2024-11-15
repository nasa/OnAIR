from onair.src.util.singleton import Singleton
from onair.src.util.service_import import import_services


class ServiceManager(Singleton):
    def __init__(self, service_dict=None):
        # Only initialize if not already initialized
        if not hasattr(self, "_initialized"):
            # Ensure service info is provided on the first instantiation
            if service_dict is None:
                raise ValueError(
                    "'service_dict' parameter required on first instantiation"
                )
            services = import_services(service_dict)
            for service_name, service in services.items():
                # Set attribute name = service name
                setattr(self, service_name, service)
            self._initialized = True  # mark as initialized to avoid re-initializing

    def get_services(self):
        # Return list of services and their functions
        services = {}
        for service_name, service in vars(self).items():
            service_funcs = set()
            if service_name.startswith("_"):
                # avoid "private" attributes, e.g. _initialized
                continue
            for f in dir(service):
                # only add the f if it's a function and not "private"
                if callable(getattr(service, f)) and not f.startswith("_"):
                    service_funcs.add(f)
            services[service_name] = service_funcs

        return services
