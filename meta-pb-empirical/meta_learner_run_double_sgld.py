import os

import learn2learn as l2l
import torch
import torch.nn as nn

from meta_learner_module_double_sgld import MetaLearnerDoubleSGLD
from utils.common import set_random_seed
from data_gen_mnist import MnistDataset

def run_meta_learner_doubleSGLD(
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
        beta,
        load_trained,
        mnist_pixels_to_permute_train=0,
        mnist_pixels_to_permute_test=0,
        seed=1):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("get tasks")
    if dataset == "mnist":
        task_sets = MnistDataset(data_path='~/data', train_shots=train_sample_size,
                                 train_ways=n_ways,
                                 test_shots=2 * n_shots,
                                 test_ways=n_ways,
                                 shuffle_pixels=True, permute_labels=False, n_pixels_to_change_train=mnist_pixels_to_permute_train,
                                 n_pixels_to_change_test=mnist_pixels_to_permute_test) #-1 shuffles all
    else:
        task_sets = l2l.vision.benchmarks.get_tasksets(
            dataset,
            train_samples=train_sample_size,
            train_ways=n_ways,
            test_samples=2 * n_shots,
            test_ways=n_ways,
            root='~/data')

    print(f"load model (dataset is {dataset})")
    if dataset == "mini-imagenet":
        model = l2l.vision.models.MiniImagenetCNN(n_ways)
    elif dataset == "omniglot":
        model = l2l.vision.models.OmniglotCNN(n_ways)
    else:
        model = l2l.vision.models.OmniglotCNN(n_ways)

    model.to(device)

    f_loss = nn.CrossEntropyLoss(reduction='mean')

    print(f"create meta learner")
    meta_learner = MetaLearnerDoubleSGLD(
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
        beta,
        reset_clf_on_meta_loop)

    model_name = f"artifacts/{dataset}/model19.pkl"

    if load_trained:
        print(f"load trained model")
        model.load_state_dict(torch.load(model_name))
    else:
        print(f"meta learner train")
        set_random_seed(seed)
        meta_learner.meta_train(n_epochs, task_sets.train)

    os.makedirs(f"artifacts/{dataset}", exist_ok=True)
    torch.save(model.state_dict(), model_name)
    import wandb
    wandb.log_artifact(model_name, name='prior-model', type='model')

    print(f"meta learner test")
    set_random_seed(seed)
    return meta_learner.meta_test(n_test_epochs, task_sets.test)
