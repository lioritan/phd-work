import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

if __name__ == "__main__":
    k = 30
    data_df = pd.read_csv(rf"C:\Users\liori\Downloads\wandb_export_stoch_{k}.csv", sep=',')
    possible_n_shots = sorted(data_df["n_shots"].unique())
    possible_n_epochs = sorted(data_df["n_test_epochs"].unique())
    possible_adapt_steps = sorted(data_df["test_adapt_steps"].unique())
    means_df = data_df.groupby(["n_shots", "n_test_epochs", "test_adapt_steps"]).mean()["test_accuracy"].reset_index()
    stderr_df = data_df.groupby(["n_shots", "n_test_epochs", "test_adapt_steps"]).sem()["test_accuracy"].reset_index()
    stderr_df.rename(columns={'test_accuracy': 'test_accuracy_stderr'}, inplace=True)

    full_df = means_df.join(stderr_df.set_index(["n_shots", "n_test_epochs", "test_adapt_steps"]),
                                           on=["n_shots", "n_test_epochs", "test_adapt_steps"])
    full_df.to_csv(rf"C:\Users\liori\Downloads\accs_stoch_{k}.csv", index=False)

    # full_df = pd.read_csv(rf"C:\Users\liori\Downloads\accs_stoch_{k}.csv", sep=',')
    # possible_n_shots = sorted(full_df["n_shots"].unique())
    # possible_n_epochs = sorted(full_df["n_test_epochs"].unique())
    # possible_adapt_steps = sorted(full_df["test_adapt_steps"].unique())

    plt.figure()
    for n_steps in possible_n_epochs:
        df_filtered = full_df[full_df["n_test_epochs"]==n_steps]
        if n_steps == 0:
            for n_epochs in possible_adapt_steps:
                # if n_epochs == 50:
                #      continue
                df_filtered2 = df_filtered[df_filtered["test_adapt_steps"]==n_epochs]
                plt.errorbar(df_filtered2["n_shots"], df_filtered2["test_accuracy"],
                             yerr=df_filtered2["test_accuracy_stderr"], label=f"Meta-testing, {n_epochs} epochs")
        else:
            if n_steps not in [5,40]:
                continue
            plt.errorbar(df_filtered["n_shots"], df_filtered["test_accuracy"],
                         yerr=df_filtered["test_accuracy_stderr"], label=f"{n_steps} adapt steps")
    plt.title(f"Test accuracies vs n_shots")
    #plt.legend(loc="lower right")
    lgd = plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1.05))
    plt.savefig(f"test_accuracies_stoch{k}.png", bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

