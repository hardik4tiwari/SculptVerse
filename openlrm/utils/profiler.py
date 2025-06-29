














from torch.profiler import profile


class DummyProfiler(profile):
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def step(self):
        pass
