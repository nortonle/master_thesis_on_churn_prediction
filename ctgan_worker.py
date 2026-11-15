# Fit a CTGAN on minority rows and emit synthetic samples.
# Run as a subprocess so the PyTorch OpenMP runtime cannot conflict with the one
# already loaded by scikit-learn / XGBoost / TensorFlow in the parent process,
# and so that a crash in here cannot bring down the notebook kernel.
import sys, numpy as np

def main():
    fin, fout, epochs, n_needed, seed = sys.argv[1:6]
    X = np.load(fin)
    np.random.seed(int(seed))
    import torch
    torch.manual_seed(int(seed))
    from ctgan import CTGAN
    import pandas as pd

    cols = [f"f{i}" for i in range(X.shape[1])]
    dfm = pd.DataFrame(X, columns=cols)
    gan = CTGAN(epochs=int(epochs), batch_size=500, verbose=False, cuda=False)
    gan.fit(dfm)
    np.save(fout, gan.sample(int(n_needed)).to_numpy(dtype=np.float32))

if __name__ == "__main__":
    main()
