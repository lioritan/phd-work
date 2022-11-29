import argparse
import wandb
import numpy as np

from meta_learner_run_pb import run_meta_learnerPB


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="mnist", choices=["mini-imagenet", "omniglot", "mnist"],
                        help="Dataset to use.")
    parser.add_argument('--train_sample_size', default=100, type=int,
                        help="Number of training examples in the inner loop at meta-train time")
    parser.add_argument('--n_ways', default=5, type=int,
                        help="Number of candidate labels (classes) at meta-test time")
    parser.add_argument('--n_shots', default=50, type=int,
                        help="Number of training examples in the inner loop at meta-test time")
    parser.add_argument('--per_task_lr', default=1e-1, type=float,
                        help="Per task LR for adaptation, should be high")
    parser.add_argument('--meta_lr', default=1e-2, type=float,
                        help="Meta LR")
    parser.add_argument('--train_adapt_steps', default=5, type=int,
                        help="Number of gradient steps to take during train adaptation")
    parser.add_argument('--test_adapt_steps', default=50, type=int,
                        help="Number of gradient steps to take during test adaptation")
    parser.add_argument('--meta_batch_size', default=8, type=int,
                        help="Number of task gradients to average for meta-gradient step")
    parser.add_argument('--n_epochs', default=100, type=int,
                        help="Meta epochs for training")
    parser.add_argument('--reset_clf_on_meta', default=False, type=bool,
                        help="Should the clf layer be reset each meta loop (should make adaptation faster)")
    parser.add_argument('--n_test_epochs', default=40, type=int,
                        help="Meta epochs for test meta-adaptation")
    parser.add_argument('--gamma', default=5000.0, type=float,
                        help="Hyper-posterior gibbs parameter")
    parser.add_argument('--beta', default=1000000.0, type=float,
                        help="Base-posterior gibbs parameter")
    parser.add_argument('--load_trained_model', default=True, type=bool,
                        help="Load pretrained model")
    parser.add_argument('--is_adaptive', default=True, type=bool,
                        help="KL adapts during run or not")
    parser.add_argument('--mnist_pixels_to_permute_train', default=100, type=int,
                        help="permutes for mnist")
    parser.add_argument('--mnist_pixels_to_permute_test', default=100, type=int,
                        help="permutes for mnist")
    parser.add_argument('--seed', type=int, default=56789, help="Random seed")
    return parser


def run_experiment(args):
    experiment_result = run_meta_learnerPB(
        dataset=args.dataset,
        train_sample_size=args.train_sample_size,
        n_ways=args.n_ways,
        n_shots=args.n_shots,
        per_task_lr=args.per_task_lr,
        meta_lr=args.meta_lr,
        train_adapt_steps=args.train_adapt_steps,
        test_adapt_steps=args.test_adapt_steps,
        meta_batch_size=args.meta_batch_size,
        n_epochs=args.n_epochs,
        reset_clf_on_meta_loop=args.reset_clf_on_meta,
        n_test_epochs=args.n_test_epochs,
        gamma=args.gamma,
        load_trained=args.load_trained_model,
        is_adaptive=args.is_adaptive,
        mnist_pixels_to_permute_train=args.mnist_pixels_to_permute_train,
        mnist_pixels_to_permute_test=args.mnist_pixels_to_permute_test,
        seed=args.seed)
    meta_error, meta_accuracy, bound_err, bound_acc, forgetting = experiment_result
    return meta_error, meta_accuracy, bound_err, bound_acc, forgetting


if __name__ == "__main__":
    args = get_parser().parse_args()
    wandb.init(project="meta-pb-stoch29l")
    wandb.config.update(args)

    if not args.load_trained_model:
        run_experiment(args)
        args.load_trained_model = True

    meta_error, meta_accuracy, bound_err, bound_acc, forgetting = run_experiment(args)
    wandb.log({"test_loss": meta_error, "test_accuracy": meta_accuracy, "bound_loss": bound_err, "bound_acc": bound_acc,
               "forgetting": forgetting, "mean_back_accuracy":1-forgetting})

    # errors = []
    # accuracies = []
    # for seed in [42, 1337, 7, 13, 999]:
    #     args.seed = seed
    #     meta_error, meta_accuracy = run_experiment(args)
    #     errors.append(meta_error)
    #     accuracies.append(meta_accuracy)
    #
    # #wandb.log({"test_loss": np.mean(errors), "test_accuracy": np.mean(accuracies)})
    # print(np.mean(accuracies))
