# TODO: Complete
class SlowModel:

    def __init__(self, model, *, source, rate=1.0, every=1):
        assert rate == 1 or rate < 0.5, rate

        self.source = source
        self.model = model
        self.rate = rate
        self.every = every
        # name = self.model.path + "_count"
        self.count = None  # nj.Variable

    def __getattr__(self, name):
        self._initonce()
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        self._initonce()
        return self.model(*args, **kwargs)

    def update(self):
        pass

    def _initonce(self, *args, method=None, **kwargs):
        pass
