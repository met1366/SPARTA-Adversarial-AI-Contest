"""
An Abstract class with utility methods that can be useful for the different defence techniques
"""

class AbstractDefence:
    def __init__(self):
        super(AbstractDefence).__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Implement in the Derived Class")

    def is_perturbed(self):
        raise NotImplementedError("To be implemented by Derived Class")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def additional_defence_loss(self, *args, **kwargs):
        raise NotImplementedError("To be implemented by Derived Class")
