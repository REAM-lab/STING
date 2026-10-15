import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compute_participation_factors(output_dir):

    A = pl.read_csv(output_dir+"/outputs/small_signal_model/A.csv")
    Amat = A[:,1:].to_numpy()
    eigvals, _ = np.linalg.eig(Amat)
    # left_eigvecs = np.linalg.inv(eigvecs).T
    # participation_factors = np.abs(left_eigvecs * eigvecs)**2

    eigenvalues, R = np.linalg.eig(Amat)   # columns of R are right eigenvectors
    L = np.linalg.inv(R)                # rows of L are left eigenvectors

    participation_factors = np.abs(L.T) * np.abs(R)
                
    # participation factors are typically normalized such that their inner product is 1 for each mode
    participation_factors = participation_factors / np.sum(participation_factors, axis=0, keepdims=True)
    state_labels = A.columns[1:]
    # make polars dataframe with rows as states and columns as modes
    pf_df = pl.DataFrame(data=participation_factors, schema=[f"Mode {i+1}" for i in range(participation_factors.shape[1])])
    # add a row to the dataframe with the eigenvalue corresponding to the mode 
    pf_df = pf_df.with_columns(pl.Series("Eig Re", [eigvals[i].real for i in range(participation_factors.shape[1])]))
    pf_df = pf_df.with_columns(pl.Series("Eig Imag", [eigvals[i].imag for i in range(participation_factors.shape[1])]))


    pf_df = pf_df.with_columns(pl.Series("State", state_labels))
    # create a new dataframe to store the maximum participation factor for each mode
    max_pf_df = pl.DataFrame({
        "Eig Re": [round(eigvals[i].real, 2) for i in range(participation_factors.shape[1])],
        "Eig Imag": [round(eigvals[i].imag, 2) for i in range(participation_factors.shape[1])],
        "Damping(%)": [round(-100*(eigvals[i].real / np.abs(eigvals[i])), 2) for i in range(participation_factors.shape[1])],
        "Freq(Hz)": [round(np.abs(eigvals[i].imag) / (2 * np.pi), 2) for i in range(participation_factors.shape[1])],
        "Max PF": [round(np.max(participation_factors[:, i]), 2) for i in range(participation_factors.shape[1])],
        "State": [state_labels[np.argmax(participation_factors[:, i])] for i in range(participation_factors.shape[1])],
    })
    # sort table by the real part of the eigenvalue
    max_pf_df_sorted = max_pf_df.sort("Eig Re")
    max_pf_df_sorted.write_csv(output_dir+"/max_participation_factors.csv")

    # make the participation factor matrix into a heatmap using seaborn

    plt.figure(figsize=(20, 15))
    sns.heatmap(participation_factors, xticklabels=[f"Mode {i+1}" for i in range(participation_factors.shape[1])], yticklabels=state_labels, cmap="viridis", annot=False, linewidths=1, linecolor='white')
    plt.title(f"Participation Factors")
    # label x ticks using the eigenvalue value, rounded to 2 dp, as well as the mode number
    plt.xticks(ticks=np.arange(participation_factors.shape[1])+0.5, labels=[f"Mode {i+1}\n{eigvals[i].real:.2f}+{eigvals[i].imag:.2f}j" for i in range(participation_factors.shape[1])])
    plt.xlabel("Modes")
    plt.ylabel("States")
    plt.tight_layout()
    plt.savefig(output_dir+f"/participation_factors_heatmap.png")
    plt.close()

    pf_df.write_csv(output_dir+"/participation_factors.csv")