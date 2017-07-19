#!/usr/bin/env python

# ==============================================================================
# This code is derived from
# https://github.com/chainer/chainer/blob/master/examples/mnist/train_mnist.py
# and partially modified by refering
# https://docs.chainer.org/en/stable/tutorial/basic.html
# ==============================================================================
# The follwoing LISENCE is copied from
# https://github.com/chainer/chainer/blob/master/LICENSE
#
# Copyright (c) 2015 Preferred Infrastructure, Inc.
# Copyright (c) 2015 Preferred Networks, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ==============================================================================


from __future__ import print_function

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions


# Network definition
class MLP(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y


def main():
    # 簡単のため以下の固定値で実行
    batchsize = 100     # 学習時のバッチサイズ
    epoch = 5          # 学習回数
    units = 1000        # 中間層の数
    gpu = -1            # GUP未使用

    print('# unit: {}'.format(units))
    print('# Minibatch-size: {}'.format(batchsize))
    print('# epoch: {}'.format(epoch))
    print('')

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    # Set data for training & data for testing (verification)
    train_iter = chainer.iterators.SerialIterator(train, batchsize,
                                                  shuffle=True)
    test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a neural network to train
    model = L.Classifier(MLP(units, 10))

    # Setup an optimizer
    # alternative: optimizer = optimizers.SGD()
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out='result')

    # Display supportive info. (evaluation of  the model, log and progress)
    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy',
                                           'validation/main/accuracy',
                                           'elapsed_time']))
    # ['epoch', 'main/loss', 'validation/main/loss',
    #  'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
