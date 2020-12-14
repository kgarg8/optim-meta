import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from gbml.gbml import GBML
from utils import get_accuracy, apply_grad, mix_grad, grad_to_cos, loss_to_ent

import pdb

class MAML(GBML):

    def __init__(self, args):
        super().__init__(args)
        self._init_net()
        self. _init_opt_MAML_native()
        return None

    def outer_loop(self, batch, is_train):

        train_inputs, train_targets, test_inputs, test_targets = self.unpack_batch(
            batch)

        test_losses = [0 for _ in range(self.args.n_inner)]
        test_corrects = [0 for _ in range(self.args.n_inner)]
        test_accs = [0 for _ in range(self.args.n_inner)]

        for (train_input, train_target, test_input, test_target) in zip(train_inputs, train_targets, test_inputs, test_targets):

            fast_weights = OrderedDict(self.network.named_parameters())

            for i in range(self.args.n_inner):
                train_logit = self.network.functional_forward(
                    train_input, fast_weights)
                train_loss = F.cross_entropy(train_logit, train_target)
                train_grad = torch.autograd.grad(
                    train_loss, fast_weights.values(), create_graph=True)

                # Update weights manually
                fast_weights = OrderedDict((name, param - self.args.inner_lr * grad)
                                           for ((name, param), grad) in zip(fast_weights.items(), train_grad))

                test_logit = self.network.functional_forward(
                    test_input, fast_weights)
                test_loss = F.cross_entropy(test_logit, test_target)
                test_loss.backward(retain_graph=True)

                test_losses[i] += test_loss
                with torch.no_grad():
                    test_acc = get_accuracy(test_logit,
                                            test_target).item()
                    test_accs[i] += test_acc

        acc_log = test_accs[-1] / self.batch_size
        loss_log = test_losses[-1] / self.args.batch_size
        if is_train:
            self.outer_optimizer.zero_grad()
            loss_log.backward()
            self.outer_optimizer.step()

            return loss_log.item(), acc_log, loss_log.item()
        else:
            return loss_log.item(), acc_log
