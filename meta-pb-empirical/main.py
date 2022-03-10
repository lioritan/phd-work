import argparse
from meta_learner_run import run_meta_learner


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="omniglot", choices=["mini-imagenet", "omniglot"],
                        help="Dataset to use.")
    parser.add_argument('--train_sample_size', default=5, type=int,
                        help="Number of training examples in the inner loop at meta-train time")
    parser.add_argument('--n_ways', default=5, type=int,
                        help="Number of candidate labels (classes) at meta-test time")
    parser.add_argument('--n_shots', default=5, type=int,
                        help="Number of training examples in the inner loop at meta-test time")
    parser.add_argument('--per_task_lr', default=1e-1, type=float,
                        help="Per task LR for adaptation, should be high")
    parser.add_argument('--meta_lr', default=1e-3, type=float,
                        help="Meta LR")
    parser.add_argument('--train_adapt_steps', default=5, type=int,
                        help="Number of gradient steps to take during train adaptation")
    parser.add_argument('--test_adapt_steps', default=10, type=int,
                        help="Number of gradient steps to take during test adaptation")
    parser.add_argument('--meta_batch_size', default=32, type=int,
                        help="Number of task gradients to average for meta-gradient step")
    parser.add_argument('--n_epochs', default=50, type=int,
                        help="Meta epochs for training")
    parser.add_argument('--seed', type=int, default=1, help="Random seed")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    run_meta_learner(
        dataset=args.dataset,
        train_sample_size=args.train_sample_size,
        n_test_labels=args.n_ways,
        n_shots=args.n_shots,
        per_task_lr=args.per_task_lr,
        meta_lr=args.meta_lr,
        train_adapt_steps=args.train_adapt_steps,
        test_adapt_steps=args.test_adapt_steps,
        meta_batch_size=args.meta_batch_size,
        n_epochs=args.n_epochs,
        seed=args.seed)
