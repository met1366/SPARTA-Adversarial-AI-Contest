"""
A base class for all the Networks created. It has some utility functions that would be useful in the framework.
"""
class Network:

    def __init__(self, name):
        super(Network, self).__init__()
        self.additional_loss = 0
        self.name = name
        self.num_classes = 0

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, *args, **kwargs):
        raise NotImplementedError

    def load(self, *args, **kwargs):
        raise NotImplementedError
