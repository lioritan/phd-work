import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_df = df = pd.read_csv(r"C:\Users\liori\Downloads\gamma_beta_1000.csv", sep=',')
    possible_n_shots = sorted(data_df["n_shots"].unique())
    # for n_shots in possible_n_shots:
    #     plt.figure()
    #     df_filtered = data_df[data_df["n_shots"] == n_shots]
    #     # plt.scatter(df_filtered["gamma"], df_filtered["beta"],
    #     #             c=df_filtered["test_accuracy"])
    #     new_df = pd.pivot_table(df_filtered, index=['beta'], columns=['gamma'])
    #     ax = sns.heatmap(new_df)
    #     ax.set(xlabel='Year', ylabel='Age Group', title='Heatmap ')
    #     plt.xscale("log")
    #     plt.yscale("log")
    #     plt.xlabel("Meta gamma (logscale)")
    #     plt.ylabel("Base beta (logscale)")
    #     plt.colorbar()
    #     plt.title(f"Test accuracies vs gamma and beta, n_shots={n_shots}")
    #     plt.savefig(f"accuray_heatmap_{n_shots}.png")
    #     plt.close()

    baselines = {2:0.777, 5:0.86 }
    baselines = {2: 0.624, 5: 0.73}

    for n_shots in possible_n_shots:
        plt.figure()
        df_filtered = data_df[data_df["n_shots"] == n_shots]
        df_filtered = df_filtered[df_filtered["beta"]==500000]
        plt.errorbar(df_filtered["gamma"], df_filtered["test_accuracy"], yerr=df_filtered["test_acc_sterr"], fmt="o")
        plt.plot(df_filtered["gamma"], baselines[n_shots]*np.ones_like(df_filtered["gamma"]), '--')
        plt.xlabel("Meta-adaptation gamma (logscale)")
        plt.xscale("log")
        plt.ylabel("Test accuracy")
        plt.ylim(bottom=np.min(df_filtered["test_accuracy"])*0.9, top=np.max(df_filtered["test_accuracy"])*1.1)
        plt.title(f"Test accuracy vs gamma, n_shots={n_shots}")
        plt.savefig(f"accuracy_plot_{n_shots}_beta.png")
        plt.close()
