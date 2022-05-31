import torch
import copy
import numpy as np
import learn2learn as l2l
from learn2learn.utils import clone_module
from learn2learn.vision.models import ConvBase

import models.stochastic_models

from optimizers.sgld_variant_optim import SimpleSGLDPriorSampling
from utils.complexity_terms import get_hyper_divergnce, get_meta_complexity_term, get_task_complexity


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def run_eval_max_posterior(model, batch, loss):
    ''' Estimates the the loss by using the mean network parameters'''
    loss_criterion = loss
    model.eval()

    inputs, targets = batch
    n_samples = len(targets)
    old_eps_std = model.set_eps_std(0.0)  # test with max-posterior
    outputs = model(inputs)
    model.set_eps_std(old_eps_std)  # return model to normal behaviour
    avg_loss = loss_criterion(outputs, targets) / n_samples
    n_correct = accuracy(outputs, targets)
    return avg_loss, n_correct


def clone_model(base_model, ctor, device):
    post_model = ctor(0).to(device)
    post_model.load_state_dict(base_model.state_dict())
    return post_model


class MetaLearnerFairPB(object):
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
            stochastic_model,
            stochastic_ctor
    ):

        self.meta_batch_size = meta_batch_size
        self.train_adapt_steps = train_adapt_steps
        self.test_adapt_steps = test_adapt_steps
        self.device = device
        self.maml = l2l.algorithms.MAML(nn_model, lr=per_task_lr, first_order=False).to(device)
        self.loss = f_loss
        self.train_opt = torch.optim.Adam(self.maml.parameters(), meta_lr)
        # self.test_opt = SimpleSGLDPriorSampling(self.maml.parameters(), meta_lr, beta=gamma)
        #self.test_opt = torch.optim.Adam(self.maml.parameters(), meta_lr)
        self.test_opt = torch.optim.Adam
        self.test_opt_params = {"lr": per_task_lr}
        self.test_opt_params = {"lr": meta_lr}
        self.seed = seed
        self.n_ways = n_ways
        self.reset_clf_on_meta_loop = reset_clf_on_meta_loop
        self.stochastic_model = stochastic_model
        self.stochastic_ctor = stochastic_ctor

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

    def split_adapt_eval(self, task_batch):
        D_task_xs, D_task_ys = task_batch
        D_task_xs, D_task_ys = D_task_xs.to(self.device), D_task_ys.to(self.device)
        task_batch_size = D_task_xs.size(0)
        # Separate data into adaptation / evaluation sets
        adapt_indices = np.zeros(task_batch_size, dtype=bool)
        train_frac = round(task_batch_size / 2)
        adapt_indices[np.arange(train_frac) * 2] = True
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

    def meta_train_pb_bound(self, n_epochs, train_taskset):
        for epoch in range(n_epochs):
            # make posterior models
            posterior_models = [clone_model(self.stochastic_model, self.stochastic_ctor, self.device)
                                for i in range(self.meta_batch_size)]
            all_post_param = sum([list(posterior_model.parameters()) for posterior_model in posterior_models], [])
            prior_params = list(self.stochastic_model.parameters())
            all_params = all_post_param + prior_params
            optimizer = self.test_opt(all_params, **self.test_opt_params) # TODO: clean code
            for step in range(self.train_adapt_steps):
                hyper_dvrg = get_hyper_divergnce(var_prior=1e2, var_posterior=1e-3, prior_model=self.stochastic_model,
                                                 device=self.device)
                meta_complex_term = get_meta_complexity_term(hyper_dvrg, delta=0.1, n_train_tasks=self.meta_batch_size)
                losses = torch.zeros(self.meta_batch_size, device=self.device)
                complexities = torch.zeros(self.meta_batch_size, device=self.device)
                for i, task in enumerate(range(self.meta_batch_size)):
                    batch = train_taskset.sample()
                    losses[i], complexities[i] = self.get_pb_terms_single_task(
                        batch[0].to(self.device), batch[1].to(self.device),
                        self.stochastic_model, posterior_models[i],
                        hyper_dvrg=hyper_dvrg, n_tasks=self.meta_batch_size)
                pb_objective = losses.mean() + complexities.mean() + meta_complex_term
                optimizer.zero_grad()
                pb_objective.backward()
                optimizer.step()
            print(f"epoch={epoch + 1}/{n_epochs}")

    def meta_test(self, test_meta_epochs, test_taskset):
        D_test_batch = test_taskset.sample()
        D_task_xs_adapt, D_task_xs_error_eval, D_task_ys_adapt, D_task_ys_error_eval = self.split_adapt_eval(
            D_test_batch)

        # Init stochastic model based on self.maml prior
        # with torch.no_grad():
        #     for elem in self.maml.features:
        #         if isinstance(elem, ConvBase):
        #             for i, block in enumerate(elem):
        #                 self.stochastic_model.convs[i].w["mean"] = block.conv.weight.clone()
        #                 self.stochastic_model.convs[i].b["mean"] = block.conv.bias.clone()
        #                 self.stochastic_model.bns[i] = copy.deepcopy(block.normalize)
        #     self.stochastic_model.fc_out.w["mean"] = self.maml.classifier.weight.clone()
        #     self.stochastic_model.fc_out.b["mean"] = self.maml.classifier.bias.clone()
        self.stochastic_model.train()

        for epoch in range(test_meta_epochs):
            # make posterior models
            posterior_models = [clone_model(self.stochastic_model, self.stochastic_ctor, self.device)
                                for i in range(self.meta_batch_size)]
            all_post_param = sum([list(posterior_model.parameters()) for posterior_model in posterior_models], [])
            prior_params = list(self.stochastic_model.parameters())
            all_params = all_post_param + prior_params
            optimizer = self.test_opt(all_params, **self.test_opt_params)
            for step in range(self.test_adapt_steps):
                hyper_dvrg = get_hyper_divergnce(var_prior=1e2, var_posterior=1e-3, prior_model=self.stochastic_model,
                                                 device=self.device)
                meta_complex_term = get_meta_complexity_term(hyper_dvrg, delta=0.1, n_train_tasks=self.meta_batch_size)
                pb_objective = self.get_pb_objective(D_task_xs_adapt, D_task_ys_adapt, hyper_dvrg,
                                                     meta_complex_term, posterior_models)
                optimizer.zero_grad()
                pb_objective.backward()
                optimizer.step()

        prior = clone_model(self.stochastic_model, self.stochastic_ctor, self.device)
        optimizer = self.test_opt(self.stochastic_model.parameters(), **self.test_opt_params)
        for step in range(self.test_adapt_steps):
            loss, complexity = self.get_pb_terms_single_task(D_task_xs_adapt, D_task_ys_adapt,
                                                             prior, self.stochastic_model)
            pb_objective = loss + complexity
            # TODO: PB data-dependant bound and optimization here!
            optimizer.zero_grad()
            pb_objective.backward()
            optimizer.step()

        # Evaluate the adapted model
        evaluation_error, evaluation_accuracy = run_eval_max_posterior(self.stochastic_model,
                                                                       (D_task_xs_error_eval, D_task_ys_error_eval),
                                                                       self.loss)

        # Logging
        print('Meta Test Error', evaluation_error.item(), flush=True)
        print('Meta Test Accuracy', evaluation_accuracy.item(), flush=True)

        return evaluation_error.item(), evaluation_accuracy.item()

    def get_pb_objective(self, x_data, y_data, hyper_dvrg, meta_complex_term, posterior_models):
        losses = torch.zeros(self.meta_batch_size, device=self.device)
        complexities = torch.zeros(self.meta_batch_size, device=self.device)
        for i, task in enumerate(range(self.meta_batch_size)):
            shuffled_indices = torch.randperm(len(y_data))
            batch = (x_data[shuffled_indices], y_data[shuffled_indices])
            D_task_xs_adapt, D_task_xs_error_eval, D_task_ys_adapt, D_task_ys_error_eval = self.split_adapt_eval(
                batch)

            losses[i], complexities[i] = self.get_pb_terms_single_task(D_task_xs_error_eval, D_task_ys_error_eval,
                                                                       self.stochastic_model, posterior_models[i],
                                                                       hyper_dvrg=hyper_dvrg,
                                                                       n_tasks=self.meta_batch_size)
        pb_objective = losses.mean() + complexities.mean() + meta_complex_term
        pb_objective = losses.mean() # TODO
        return pb_objective

    def get_pb_terms_single_task(self, x, y, prior, posterior, hyper_dvrg=0, n_tasks=1):
        n_MC = 1  # TODO: move hyper-params
        avg_empiric_loss = 0.0
        n_samples = len(y)
        for i_MC in range(n_MC):
            # Empirical Loss on current task:
            outputs = posterior(x)
            avg_empiric_loss_curr = (1 / n_samples) * self.loss(outputs, y)
            avg_empiric_loss += (1 / n_MC) * avg_empiric_loss_curr

        # TODO: we can optimize the bound using x_adapt, y_adapt...
        complexity = get_task_complexity(bound_type="McAllester", delta=0.1, kappa_post=1e-3,
                                         prior_model=prior, post_model=posterior,
                                         n_samples=n_samples, avg_empiric_loss=avg_empiric_loss, hyper_dvrg=hyper_dvrg,
                                         n_train_tasks=n_tasks, noised_prior=True)
        return avg_empiric_loss, complexity
