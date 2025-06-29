














class Registry:
    """Registry class"""

    def __init__(self):
        self._registry = {}

    def register(self, name):
        """Register a module"""
        def decorator(cls):
            assert name not in self._registry, 'Module {} already registered'.format(name)
            self._registry[name] = cls
            return cls
        return decorator

    def __getitem__(self, name):
        """Get a module"""
        return self._registry[name]

    def __contains__(self, name):
        return name in self._registry
