class HFtqdm:
    def __init__(self, *args, **kwargs):
        self.total = kwargs.get("total")
    def update(self, *args, **kwargs):
        pass
    def close(self):
        pass

tqdm = HFtqdm
