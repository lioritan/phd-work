import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

if __name__ == "__main__":
    comparison_types = {"baseline":"meta-testing", "nonadapt":"AML, constant gamma", "adaptive_fixed":"AML, adaptive gamma",
                        "adaptive_sgd_10meta":"AML, adaptive, SGD base learner"}#, "adaptive_low_start_gamma", "adaptive_longer"]
    model_name_dict = {19: "misspecified", 21:"aligned"}
    ctype_dfs = []
    for ctype in comparison_types.keys():
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
        model_name = model_name_dict[model_num]
        plt.figure()
        for i, ctype in enumerate(comparison_types.keys()):
            df_filtered = ctype_dfs[i][ctype_dfs[i]["model_num"]==model_num]
            plt.errorbar(df_filtered["n_shots"], df_filtered["test_accuracy"], yerr=df_filtered["test_accuracy_stderr"], label=f"{comparison_types[ctype]}")
        plt.title(f"Test accuracies vs n_shots, prior type: {model_name}")
        plt.legend(loc="lower right")
        plt.savefig(f"test_accuracies_model_{model_name}.png")
        plt.close()

