import torch.nn as nn


class PrintLayer(nn.Module):
    # use this class to debug in nn.Sequential
    def __init__(self, display="shape", quit_after=False):
        super(PrintLayer, self).__init__()
        self.display = display
        self.quit_after = quit_after

    def forward(self, x):
        if self.display == "shape":
            print(x.shape)
        else:
            print(x)
        if self.quit_after:
            quit()
        return x
