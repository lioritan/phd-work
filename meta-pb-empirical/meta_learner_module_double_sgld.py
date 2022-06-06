import math

import learn2learn as l2l
import numpy as np
import torch
from learn2learn.utils import update_module
from torch.autograd import grad

from optimizers.sgld_variant_optim import SimpleSGLDPriorSampling


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def clone_model(base_model, ctor, device):
    post_model = ctor(0).to(device)
    post_model.load_state_dict(base_model.state_dict())
    return post_model


class MetaLearnerDoubleSGLD(object):
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
            beta,
            reset_clf_on_meta_loop,
            shots_mult,
    ):

        self.meta_batch_size = meta_batch_size
        self.train_adapt_steps = train_adapt_steps
        self.test_adapt_steps = test_adapt_steps
        self.device = device
        self.maml = l2l.algorithms.MAML(nn_model, lr=per_task_lr, first_order=False).to(device)
        self.loss = f_loss
        self.train_opt = torch.optim.Adam(self.maml.parameters(), meta_lr)
        self.test_opt = SimpleSGLDPriorSampling(self.maml.parameters(), meta_lr, beta=gamma)
        self.test_beta = beta
        self.seed = seed
        self.n_ways = n_ways
        self.reset_clf_on_meta_loop = reset_clf_on_meta_loop
        self.shots_mult = shots_mult
        self.adaptive_gamma = True

    def calculate_meta_loss(self, task_batch, learner, adapt_steps, training_mode=True):
        split_data = self.split_adapt_eval(task_batch)
        return self.adapt_model(split_data, learner, adapt_steps, training_mode)

    def manual_grad_sgld(self, learner, error):
        diff_params = [p for p in learner.module.parameters() if p.requires_grad]
        grad_params = grad(error,
                           diff_params,
                           retain_graph=True,
                           create_graph=True,
                           allow_unused=False)
        gradients = []
        grad_counter = 0

        # Handles gradients for non-differentiable parameters
        for param in learner.module.parameters():
            if param.requires_grad:
                gradient = grad_params[grad_counter]
                grad_counter += 1
            else:
                gradient = None
            gradients.append(gradient)
        params = list(learner.module.parameters())
        if not len(gradients) == len(diff_params):
            msg = 'WARNING:maml_update(): Parameters and gradients have different length. ('
            msg += str(len(params)) + ' vs ' + str(len(gradients)) + ')'
            print(msg)
        pass
        # if beta is high, we want posterior sampling, if it's low we want prior sampling
        noise_variance = math.sqrt(self.maml.lr / self.test_beta) if self.test_beta >= 1.0 else math.sqrt(
            self.test_beta)
        scaled_noise = torch.normal(
            mean=torch.zeros(len(gradients)),
            std=torch.ones(len(gradients)) * noise_variance
        ).to(self.device)
        for p, g, xi in zip(params, gradients, scaled_noise):
            if g is not None:
                if self.test_beta >= 1.0:
                    p.update = - self.maml.lr * g + xi
                else:
                    p.update = xi
        update_module(learner.module)

    def adapt_model(self, task_batch, learner, adapt_steps, training_mode=True):
        D_task_xs_adapt, D_task_xs_error_eval, D_task_ys_adapt, D_task_ys_error_eval = task_batch

        # Adapt the model
        for step in range(adapt_steps):
            adaptation_error = self.loss(learner(D_task_xs_adapt), D_task_ys_adapt)
            if training_mode:
                learner.adapt(adaptation_error)
            else:
                self.manual_grad_sgld(learner, adaptation_error)

        # Evaluate the adapted model
        predictions = learner(D_task_xs_error_eval)
        evaluation_error = self.loss(predictions, D_task_ys_error_eval)
        evaluation_accuracy = accuracy(predictions, D_task_ys_error_eval)

        return evaluation_error, evaluation_accuracy

    def split_adapt_eval(self, task_batch, train_frac=2):
        D_task_xs, D_task_ys = task_batch
        D_task_xs, D_task_ys = D_task_xs.to(self.device), D_task_ys.to(self.device)
        task_batch_size = D_task_xs.size(0)
        # Separate data into adaptation / evaluation sets - works even if labels are ordered
        adapt_indices = np.zeros(task_batch_size, dtype=bool)
        train_samples = round(task_batch_size / train_frac)
        adapt_indices[np.arange(train_samples) * train_frac] = True
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
                    self.calculate_meta_loss(batch, learner, self.train_adapt_steps, training_mode=True)

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
        D_task_xs_adapt, D_task_xs_error_eval, D_task_ys_adapt, D_task_ys_error_eval = self.split_adapt_eval(
            D_test_batch, train_frac=self.shots_mult)

        if self.adaptive_gamma:
            non_adapt_loss = self.get_base_learner_loss(D_task_xs_adapt, D_task_xs_error_eval,
                                                        D_task_ys_adapt, D_task_ys_error_eval)
            loss_diff = 0
            last_adapt_loss = non_adapt_loss
            old_gamma = self.test_opt.param_groups[0]["beta"]
            dw_dgamma = None

        # TODO: consider doing this without subsampling since both levels are already randomized - use adapt_model
        for epoch in range(test_meta_epochs):
            self.test_opt.zero_grad()
            for task in range(self.meta_batch_size):
                # Compute meta-training loss
                learner = self.maml.clone().to(self.device)
                # shuffle and meta-adapt (divide half-half)
                shuffled_indices = torch.randperm(len(D_task_ys_adapt))
                batch = (D_task_xs_adapt[shuffled_indices], D_task_ys_adapt[shuffled_indices])
                # batch = (D_task_xs_adapt, D_task_ys_adapt)
                evaluation_error, evaluation_accuracy = \
                    self.calculate_meta_loss(batch, learner, self.test_adapt_steps, training_mode=False)
                evaluation_error.backward()

            # Average the accumulated task gradients and optimize
            for p in self.maml.parameters():  # Note: this is somewhat bad practice
                p.grad.data.mul_(1.0 / self.meta_batch_size)
            self.test_opt.step()

            if self.adaptive_gamma:
                adapt_loss = self.get_base_learner_loss(D_task_xs_adapt, D_task_xs_error_eval,
                                                        D_task_ys_adapt, D_task_ys_error_eval)
                with torch.no_grad():
                    new_loss_diff = adapt_loss - non_adapt_loss
                    n_params = len(self.test_opt.last_noise)
                    dl_dw = [param.grad.data for param in self.test_opt.param_groups[0]["params"] if
                             param.requires_grad]
                    dl_dgamma = 0
                    if dw_dgamma is None:
                        dw_dgamma = [0 for i in range(n_params)]
                    for param_ind in range(n_params):
                        dw_dgamma[param_ind] = dw_dgamma[param_ind] + self.test_opt.last_noise[param_ind] * (
                                    -0.5 / old_gamma)
                        dl_dgamma += torch.sum(dw_dgamma[param_ind] * dl_dw[param_ind])
                    if dl_dgamma == 0:
                        dl_dgamma = old_gamma
                    new_gamma = new_loss_diff / dl_dgamma
                    old_gamma = self.test_opt.param_groups[0]["beta"]
                    if new_gamma > 1.0:
                        self.test_opt.param_groups[0]["beta"] = 5.0 + new_gamma
                    elif new_gamma < -1.0:
                        self.test_opt.param_groups[0]["beta"] = 5.0 - new_gamma
                    # else:
                    #     self.test_opt.param_groups[0]["beta"] = new_gamma

        learner = self.maml.clone()
        evaluation_error, evaluation_accuracy = self.adapt_model(
            (D_task_xs_adapt, D_task_xs_error_eval, D_task_ys_adapt, D_task_ys_error_eval),
            learner, self.test_adapt_steps, training_mode=False)

        # Logging
        print('Meta Test Error', evaluation_error.item(), flush=True)
        print('Meta Test Accuracy', evaluation_accuracy.item(), flush=True)

        return evaluation_error.item(), evaluation_accuracy.item()

    def get_base_learner_loss(self, D_task_xs_adapt, D_task_xs_error_eval, D_task_ys_adapt, D_task_ys_error_eval):
        learner = self.maml.clone()
        evaluation_error, _ = self.adapt_model(
            (D_task_xs_adapt, D_task_xs_error_eval, D_task_ys_adapt, D_task_ys_error_eval),
            learner, self.test_adapt_steps, training_mode=False)
        del learner
        return evaluation_error
