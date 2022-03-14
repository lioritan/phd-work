import os

import learn2learn as l2l
import torch
import torch.nn as nn

from Utils.common import set_random_seed
from meta_learner_module import MetaLearner


def run_meta_learner(
        dataset,
        train_sample_size,
        n_ways,
        n_shots,
        per_task_lr,
        meta_lr,
        train_adapt_steps,
        test_adapt_steps,
        meta_batch_size,
        n_epochs,
        reset_clf_on_meta_loop,
        n_test_epochs,
        gamma,
        load_trained,
        seed=1):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("get tasks")
    task_sets = l2l.vision.benchmarks.get_tasksets(
        dataset,
        train_samples=train_sample_size,
        train_ways=n_ways,
        test_samples=2 * n_shots,
        test_ways=n_ways,
        root='~/data')

    # TODO: change this, consider the stochastic network case
    print(f"load model (dataset is {dataset})")
    if dataset == "mini-imagenet":
        model = l2l.vision.models.MiniImagenetCNN(n_ways)
    else:
        model = l2l.vision.models.OmniglotCNN(n_ways)
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
        seed,
        n_ways,
        gamma,
        reset_clf_on_meta_loop)

    if load_trained:
        print(f"load trained model")
        model.load_state_dict(torch.load("artifacts/model.pkl"))
    else:
        print(f"meta learner train")
        set_random_seed(seed)
        meta_learner.meta_train(n_epochs, task_sets.train)

    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/model.pkl")

    print(f"meta learner test")
    set_random_seed(seed)
    meta_learner.meta_test(n_test_epochs, task_sets.test)
