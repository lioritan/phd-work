import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_df = df = pd.read_csv(r"C:\Users\liori\Downloads\experiment_data.csv", sep=',')
    possible_n_shots = sorted(data_df["n_shots"].unique())
    for n_shots in possible_n_shots:
        plt.figure()
        df_filtered = data_df[data_df["n_shots"] == n_shots]
        interesting_beta = df_filtered["beta"] > 1000
        plt.scatter(np.log10(df_filtered["gamma"][interesting_beta]), np.log10(df_filtered["beta"][interesting_beta]),
                    c=df_filtered["test_accuracy"][interesting_beta])
        plt.xlabel("Meta gamma (logscale 10)")
        plt.ylabel("Base beta (logscale 10)")
        plt.colorbar()
        plt.title(f"Test accuracies vs gamma and beta, n_shots={n_shots}")
        plt.savefig(f"test_accuracies_{n_shots}.png")
        plt.close()
