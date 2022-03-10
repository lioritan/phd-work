import learn2learn as l2l
import torch
import torch.nn as nn

from Utils.common import set_random_seed
from meta_learner_module import MetaLearner


def run_meta_learner(
        dataset,
        train_sample_size,
        n_test_labels,
        n_shots,
        per_task_lr,
        meta_lr,
        train_adapt_steps,
        test_adapt_steps,
        meta_batch_size,
        n_epochs,
        seed=1):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("get tasks")
    task_sets = l2l.vision.benchmarks.get_tasksets(
        dataset,
        train_samples=train_sample_size,
        train_ways=n_test_labels,
        test_samples=2 * n_shots,
        test_ways=n_test_labels,
        root='~/data')

    # TODO: change this, consider the stochastic network case
    print(f"load model (dataset is {dataset})")
    if dataset == "mini-imagenet":
        model = l2l.vision.models.MiniImagenetCNN(n_test_labels)
    else:
        model = l2l.vision.models.OmniglotCNN(n_test_labels)
    model.to(device)

    f_loss = nn.CrossEntropyLoss(reduction='mean')

    print(f"create meta learner")
    meta_learner = MetaLearner(
        per_task_lr,
        meta_lr,
        train_adapt_steps,
        test_adapt_steps,
        meta_batch_size,
        model,
        f_loss,
        device,
        seed)

    print(f"meta learner train")
    set_random_seed(seed)
    meta_learner.meta_train(n_epochs, task_sets.train)

    print(f"meta learner test")
    set_random_seed(seed)
    meta_learner.meta_test(task_sets.test)
