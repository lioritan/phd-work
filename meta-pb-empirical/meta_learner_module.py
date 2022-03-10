import torch
import numpy as np
import learn2learn as l2l


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


class MetaLearner(object):
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
            seed):

        self.meta_batch_size = meta_batch_size
        self.train_adapt_steps = train_adapt_steps
        self.test_adapt_steps = test_adapt_steps
        self.device = device
        self.maml = l2l.algorithms.MAML(nn_model, lr=per_task_lr, first_order=False).to(device)
        self.loss = f_loss
        self.opt = torch.optim.Adam(self.maml.parameters(), meta_lr)
        self.seed = seed

    def calculate_meta_loss(self, task_batch, learner, adapt_steps):
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

        # Adapt the model
        for step in range(adapt_steps):
            adaptation_error = self.loss(learner(D_task_xs_adapt), D_task_ys_adapt)
            learner.adapt(adaptation_error)

        # Evaluate the adapted model
        predictions = learner(D_task_xs_error_eval)
        features = learner.module.features(D_task_xs_error_eval)
        evaluation_error = self.loss(predictions, D_task_ys_error_eval)
        evaluation_accuracy = accuracy(predictions, D_task_ys_error_eval)

        del D_task_xs_error_eval, D_task_xs_adapt

        return evaluation_error, evaluation_accuracy, predictions, features

    def meta_train(self, n_epochs, train_taskset):

        meta_train_errors = []
        meta_train_accs = []

        for epoch in range(n_epochs):

            self.opt.zero_grad()
            meta_train_error = 0.0
            meta_train_accuracy = 0.0

            for task in range(self.meta_batch_size):
                # Compute meta-training loss
                learner = self.maml.clone().to(self.device)
                # get task from schedule
                batch = train_taskset.sample()
                evaluation_error, evaluation_accuracy, prediction, features = \
                    self.calculate_meta_loss(batch, learner, self.train_adapt_steps)

                evaluation_error.backward() # TODO: SGLD
                meta_train_error += evaluation_error.item()
                meta_train_accuracy += evaluation_accuracy.item()

            # Average the accumulated task gradients and optimize
            for p in self.maml.parameters():  # Note: this is somewhat bad practice
                p.grad.data.mul_(1.0 / self.meta_batch_size)
            self.opt.step()
            # Logging
            norm_meta_train_error = meta_train_error / self.meta_batch_size
            norm_meta_train_accuracy = meta_train_accuracy / self.meta_batch_size
            meta_train_errors.append(norm_meta_train_error)
            meta_train_accs.append(norm_meta_train_accuracy)
            print(f"epoch={epoch + 1}/{n_epochs}, "
                  f"loss={norm_meta_train_error:.3f}, "
                  f"acc={norm_meta_train_accuracy:.3f}")

        return meta_train_errors, meta_train_accs

    def meta_test(self, test_taskset):

        # calculate test error

        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        for task in range(self.meta_batch_size):
            # Compute meta-testing loss
            learner = self.maml.clone()
            # Random sampling
            D_test_batch = test_taskset.sample()
            evaluation_error, evaluation_accuracy, _, _ = \
                self.calculate_meta_loss(D_test_batch, learner, self.test_adapt_steps)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()

        norm_meta_test_error = meta_test_error / self.meta_batch_size
        norm_meta_test_accuracy = meta_test_accuracy / self.meta_batch_size
        # Logging
        print('Meta Test Error', norm_meta_test_error, flush=True)
        print('Meta Test Accuracy', norm_meta_test_accuracy, flush=True)

        return norm_meta_test_error, norm_meta_test_accuracy
