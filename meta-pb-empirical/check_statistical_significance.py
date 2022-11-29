from scipy.stats import wilcoxon
import pandas as pd
import numpy as np

if __name__ == "__main__":
    pixels_raws = [r"C:\Users\liori\Downloads\wandb_export_stoch_17.csv", r"C:\Users\liori\Downloads\wandb_export_stoch_22.csv"]
    labels_raws = [r"C:\Users\liori\Downloads\wandb_export_stoch_23.csv", r"C:\Users\liori\Downloads\wandb_export_stoch_26.csv"]

    shots_vals = [2,5,10,20,50]
    # pixels_df_const = pd.read_csv(pixels_raws[0])
    # pixels_df_const = pixels_df_const[pixels_df_const["n_shots"].isin(shots_vals)]
    # pixels_df_const = pixels_df_const[pixels_df_const["n_test_epochs"].isin([0, 40])]
    # pixels_baseline = pixels_df_const[(pixels_df_const["n_test_epochs"]==0) & (pixels_df_const["test_adapt_steps"]==1050)]
    # pixels_baseline.sort_values(by="seed", inplace=True)
    #
    # pixels_const = pixels_df_const[pixels_df_const["n_test_epochs"]==40]
    # pixels_const.sort_values(by="seed", inplace=True)
    #
    # pixels_df_adapt = pd.read_csv(pixels_raws[1])
    # pixels_df_adapt = pixels_df_adapt[pixels_df_adapt["n_shots"].isin(shots_vals)]
    # pixels_df_adapt = pixels_df_adapt[pixels_df_adapt["n_test_epochs"]==40]
    # pixels_df_adapt.sort_values(by="seed", inplace=True)
    #
    # for shots in shots_vals:
    #     # w_res = wilcoxon(pixels_baseline[pixels_baseline["n_shots"]==shots]["test_accuracy"],
    #     #                  pixels_const[pixels_const["n_shots"]==shots]["test_accuracy"])
    #     # print(f"Constant prior, {shots} shots: {w_res.pvalue}")
    #
    #     w_res = wilcoxon(pixels_baseline[pixels_baseline["n_shots"] == shots]["test_accuracy"],
    #                      pixels_df_adapt[pixels_df_adapt["n_shots"] == shots]["test_accuracy"])
    #     print(f"Adaptive prior, {shots} shots: {w_res.pvalue}")

    labels_df_const = pd.read_csv(labels_raws[0])
    labels_df_const = labels_df_const[labels_df_const["n_shots"].isin(shots_vals)]
    labels_df_const = labels_df_const[labels_df_const["n_test_epochs"].isin([0, 40])]
    labels_baseline = labels_df_const[
        (labels_df_const["n_test_epochs"] == 0) & (labels_df_const["test_adapt_steps"] == 1000)]
    labels_baseline.sort_values(by="seed", inplace=True)

    labels_const = labels_df_const[labels_df_const["n_test_epochs"] == 40]
    labels_const.sort_values(by="seed", inplace=True)

    labels_df_adapt = pd.read_csv(labels_raws[1])
    labels_df_adapt = labels_df_adapt[labels_df_adapt["n_shots"].isin(shots_vals)]
    labels_df_adapt = labels_df_adapt[labels_df_adapt["n_test_epochs"] == 40]
    labels_df_adapt.sort_values(by="seed", inplace=True)

    for shots in shots_vals:
        w_res = wilcoxon(labels_baseline[labels_baseline["n_shots"]==shots]["test_accuracy"],
                         labels_const[labels_const["n_shots"]==shots]["test_accuracy"])
        print(f"Constant prior, {shots} shots: {w_res.pvalue}")

        # if shots == 2:
        #     continue
        # w_res = wilcoxon(labels_baseline[labels_baseline["n_shots"] == shots]["test_accuracy"],
        #                  labels_df_adapt[labels_df_adapt["n_shots"] == shots]["test_accuracy"])
        # print(f"Adaptive prior, {shots} shots: {w_res.pvalue}")


