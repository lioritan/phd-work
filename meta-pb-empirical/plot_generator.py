import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

if __name__ == "__main__":
    comparison_types = ["baseline", "nonadapt", "adaptive_longer", "adaptive_fixed", "adaptive_low_start_gamma"]
    ctype_dfs = []
    for ctype in comparison_types:
        data_df = df = pd.read_csv(rf"C:\Users\liori\Downloads\all_models_{ctype}.csv", sep=',')
        possible_n_shots = sorted(data_df["n_shots"].unique())
        possible_models = sorted(data_df["model_num"].unique())
        means_df = data_df.groupby(["n_shots", "beta", "gamma", "model_num"]).mean()["test_accuracy"].reset_index()

        best_params_df = data_df.groupby(["n_shots", "model_num"]).max().reset_index()
        stderr_df = data_df.groupby(["n_shots", "beta", "gamma", "model_num"]).sem()["test_accuracy"].reset_index()
        stderr_df.rename(columns={'test_accuracy': 'test_accuracy_stderr'}, inplace=True)

        filtered_stderrs = best_params_df.join(stderr_df.set_index(["n_shots", "beta", "gamma", "model_num"]), on=["n_shots", "beta", "gamma", "model_num"])
        filtered_stderrs.to_csv(rf"C:\Users\liori\Downloads\accs_all_models_{ctype}.csv", index=False)
        ctype_dfs.append(filtered_stderrs)

    for model_num in possible_models:
        plt.figure()
        for i, ctype in enumerate(comparison_types):
            df_filtered = ctype_dfs[i][ctype_dfs[i]["model_num"]==model_num]
            plt.errorbar(df_filtered["n_shots"], df_filtered["test_accuracy"], yerr=df_filtered["test_accuracy_stderr"], label=f"{ctype}")
        plt.title(f"Test accuracies vs n_shots, model={model_num}")
        plt.legend()
        plt.savefig(f"test_accuracies_model {model_num}.png")
        plt.close()

