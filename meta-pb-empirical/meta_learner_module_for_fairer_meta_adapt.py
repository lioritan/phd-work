import torch
import numpy as np
import learn2learn as l2l
from learn2learn.utils import clone_module

from optimizers.sgld_variant_optim import SimpleSGLDPriorSampling


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


class MetaLearnerFair(object):
    def __init__(
            self,
            per_task_lr,
            meta_lr,
            train_adapt_steps,
            test_adapt_steps,
            meta_batch_size,
            nn_model,
            f_loss,
            device,
            seed,
            n_ways,
            gamma,
            reset_clf_on_meta_loop,
    ):

        self.meta_batch_size = meta_batch_size
        self.train_adapt_steps = train_adapt_steps
        self.test_adapt_steps = test_adapt_steps
        self.device = device
        self.maml = l2l.algorithms.MAML(nn_model, lr=per_task_lr, first_order=False).to(device)
        self.loss = f_loss
        self.train_opt = torch.optim.Adam(self.maml.parameters(), meta_lr)
        self.test_opt = SimpleSGLDPriorSampling(self.maml.parameters(), meta_lr, beta=gamma)
        self.seed = seed
        self.n_ways = n_ways
        self.reset_clf_on_meta_loop = reset_clf_on_meta_loop

    def calculate_meta_loss(self, task_batch, learner, adapt_steps):
        D_task_xs_adapt, D_task_xs_error_eval, D_task_ys_adapt, D_task_ys_error_eval = self.split_adapt_eval(task_batch)

        # Adapt the model
        for step in range(adapt_steps):
            adaptation_error = self.loss(learner(D_task_xs_adapt), D_task_ys_adapt)
            learner.adapt(adaptation_error)

        # Evaluate the adapted model
        predictions = learner(D_task_xs_error_eval)
        evaluation_error = self.loss(predictions, D_task_ys_error_eval)
        evaluation_accuracy = accuracy(predictions, D_task_ys_error_eval)

        del D_task_xs_error_eval, D_task_xs_adapt

        return evaluation_error, evaluation_accuracy

    def split_adapt_eval(self, task_batch, div=2):
        D_task_xs, D_task_ys = task_batch
        D_task_xs, D_task_ys = D_task_xs.to(self.device), D_task_ys.to(self.device)
        task_batch_size = D_task_xs.size(0)
        # Separate data into adaptation / evaluation sets
        adapt_indices = np.zeros(task_batch_size, dtype=bool)
        train_frac = round(task_batch_size / div)
        adapt_indices[np.arange(train_frac) * div] = True
        error_eval_indices = ~adapt_indices
        # numpy -> torch
        adapt_indices = torch.from_numpy(adapt_indices)
        error_eval_indices = torch.from_numpy(error_eval_indices)
        D_task_xs_adapt, D_task_ys_adapt = D_task_xs[adapt_indices], D_task_ys[adapt_indices]
        D_task_xs_error_eval, D_task_ys_error_eval = D_task_xs[error_eval_indices], D_task_ys[error_eval_indices]
        return D_task_xs_adapt, D_task_xs_error_eval, D_task_ys_adapt, D_task_ys_error_eval

    def meta_train(self, n_epochs, train_taskset):
        opt = self.train_opt

        meta_train_errors = []
        meta_train_accs = []

        for epoch in range(n_epochs):
            opt.zero_grad()
            if self.reset_clf_on_meta_loop:
                self.maml.module.classifier.weight.data.normal_()
                self.maml.module.classifier.bias.data.mul_(0.0)
            meta_train_error = 0.0
            meta_train_accuracy = 0.0

            for task in range(self.meta_batch_size):
                # Compute meta-training loss
                learner = self.maml.clone().to(self.device)
                # sample
                batch = train_taskset.sample()
                evaluation_error, evaluation_accuracy = \
                    self.calculate_meta_loss(batch, learner, self.train_adapt_steps)

                evaluation_error.backward()
                meta_train_error += evaluation_error.item()
                meta_train_accuracy += evaluation_accuracy.item()

            # Average the accumulated task gradients and optimize
            for p in self.maml.parameters():  # Note: this is somewhat bad practice
                p.grad.data.mul_(1.0 / self.meta_batch_size)
            opt.step()
            # Logging
            norm_meta_train_error = meta_train_error / self.meta_batch_size
            norm_meta_train_accuracy = meta_train_accuracy / self.meta_batch_size
            meta_train_errors.append(norm_meta_train_error)
            meta_train_accs.append(norm_meta_train_accuracy)
            print(f"epoch={epoch + 1}/{n_epochs}, "
                  f"loss={norm_meta_train_error:.3f}, "
                  f"acc={norm_meta_train_accuracy:.3f}")

        return meta_train_errors, meta_train_accs

    def meta_test(self, test_meta_epochs, test_taskset):
        D_test_batch = test_taskset.sample()
        D_task_xs_adapt, D_task_xs_error_eval, D_task_ys_adapt, D_task_ys_error_eval = self.split_adapt_eval(D_test_batch, div=10)

        for epoch in range(test_meta_epochs):
            self.test_opt.zero_grad()
            for task in range(self.meta_batch_size):
                # Compute meta-training loss
                learner = self.maml.clone().to(self.device)
                # shuffle and meta-adapt (divide half-half)
                shuffled_indices = torch.randperm(len(D_task_ys_adapt))
                batch = (D_task_xs_adapt[shuffled_indices], D_task_ys_adapt[shuffled_indices])
                evaluation_error, evaluation_accuracy = \
                    self.calculate_meta_loss(batch, learner, self.test_adapt_steps)
                evaluation_error.backward()

            # Average the accumulated task gradients and optimize
            for p in self.maml.parameters():  # Note: this is somewhat bad practice
                p.grad.data.mul_(1.0 / self.meta_batch_size)
            self.test_opt.step()

        learner = self.maml.clone()
        for step in range(self.test_adapt_steps):
            adaptation_error = self.loss(learner(D_task_xs_adapt), D_task_ys_adapt)
            learner.adapt(adaptation_error)

        # Evaluate the adapted model
        predictions = learner(D_task_xs_error_eval)
        evaluation_error = self.loss(predictions, D_task_ys_error_eval)
        evaluation_accuracy = accuracy(predictions, D_task_ys_error_eval)

        # Logging
        print('Meta Test Error', evaluation_error.item(), flush=True)
        print('Meta Test Accuracy', evaluation_accuracy.item(), flush=True)

        return evaluation_error.item(), evaluation_accuracy.item()
