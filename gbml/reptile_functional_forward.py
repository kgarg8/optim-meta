import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from gbml.gbml import GBML
from utils import get_accuracy, apply_grad, mix_grad, grad_to_cos, loss_to_ent
from copy import deepcopy
import pdb

class Reptile(GBML):

    def __init__(self, args):
        super().__init__(args)
        self._init_net()
        self._init_opt()
        return None

    def outer_loop(self, batch, is_train):

        train_inputs, train_targets, test_inputs, test_targets = self.unpack_batch(
            batch)

        test_losses = [0 for _ in range(self.args.n_inner)]
        test_corrects = [0 for _ in range(self.args.n_inner)]
        test_accs = [0 for _ in range(self.args.n_inner)]

        new_weights = []
        weights_original = deepcopy(self.network.state_dict())
        for (train_input, train_target, test_input, test_target) in zip(train_inputs, train_targets, test_inputs, test_targets):

            fast_weights = OrderedDict(self.network.named_parameters())

            # inner loop
            for i in range(self.args.n_inner):
                train_logit = self.network.functional_forward(
                    train_input, fast_weights)
                train_loss = F.cross_entropy(train_logit, train_target)
                train_grad = torch.autograd.grad(
                    train_loss, fast_weights.values())

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

            # new weights for the current batch
            new_weights.append(fast_weights)
            # restore the network to original weights before processing the next batch
            self.network.load_state_dict(
                {name: weights_original[name] for name in weights_original})

        acc_log = test_accs[-1] / self.batch_size
        loss_log = test_losses[-1] / self.args.batch_size
        if is_train:
            ws = len(new_weights)
            fast_weights = {
                name: new_weights[0][name] / float(ws) for name in new_weights[0]}
            for i in range(1, ws):
                for name in new_weights[i]:
                    fast_weights[name] += new_weights[i][name] / float(ws)

            named_params = [name for name, param in fast_weights.items()]
            for name, param in weights_original.items():
                if name in named_params:
                    weights_original[name] = weights_original[name] + (
                        (fast_weights[name] - weights_original[name]) * self.args.outer_lr)
            self.network.load_state_dict(weights_original)
            # self.network.load_state_dict({name: weights_original[name] + (
            #     (fast_weights[name] - weights_original[name]) * self.args.outer_lr) for name in weights_original})

            return loss_log.item(), acc_log, loss_log.item()
        else:
            return loss_log.item(), acc_log
