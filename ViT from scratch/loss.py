try:
    import cupy as cp
except ImportError:
    import numpy as cp
class CrossEntropy():
    def __init__(self, padding_id=-1, vocab_size=100, label_smoothing=0.0):
        self.eps = 1e-6
        self.padding_id = padding_id
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing

    def one_hot(self, label):
        label = cp.asarray(label)
        label_num = len(label)
        one_hot = cp.zeros((label_num, self.vocab_size))
        one_hot[cp.arange(label_num), label] = 1.0
        if self.label_smoothing > 0.0:
            smooth_val = float(self.label_smoothing) / float(self.vocab_size)
            one_hot = one_hot * (1.0 - self.label_smoothing) + smooth_val
        self.one_hot_label = one_hot
        return one_hot

    def forward(self, pred, label):
        one_hot_label = self.one_hot(label)
        loss = -(one_hot_label * cp.log(pred + self.eps)).sum(-1)
        return loss

    def grad(self, pred, label):
        batch_size = pred.shape[0]
        grad = (pred - self.one_hot_label) / batch_size
        return grad

class MSE():
    def __init__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

    def forward(self):
        loss = ((self.y_pred - self.y_true) ** 2).mean()
        return loss

    def grad(self):
        grad = 2 * (self.y_pred - self.y_true)
        return grad
