import pandas as pd
import numpy as np

if __name__ == "__main__":
    pixels_df = pd.read_csv(r"C:\Users\liori\Downloads\wandb_export_bounds_pixel.csv")

    possible_n_shots = sorted(pixels_df["n_shots"].unique())
    possible_n_epochs = sorted(pixels_df["n_test_epochs"].unique())
    possible_adaptive = sorted(pixels_df["is_adaptive"].unique())
    pixels_df.fillna(0, inplace=True)
    means_df = pixels_df.groupby(["n_shots", "n_test_epochs", "is_adaptive"]).mean()["bound_acc"].reset_index()

    full_df = means_df[means_df["n_test_epochs"].isin([0,40])]

    print(full_df)
