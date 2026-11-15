# Fit a CTGAN on minority rows and emit synthetic samples.
# Run as a subprocess so PyTorch's OpenMP runtime cannot conflict with the one
# already loaded by scikit-learn / XGBoost / TensorFlow in the parent process,
# and so a crash in here cannot bring down the notebook kernel.
import sys
import numpy as np


def main():
    path_in, path_out, epochs, n_needed, seed = sys.argv[1:6]
    X = np.load(path_in)

    np.random.seed(int(seed))
    import torch
    torch.manual_seed(int(seed))

    import pandas as pd
    from ctgan import CTGAN

    columns = [f"f{i}" for i in range(X.shape[1])]
    generator = CTGAN(epochs=int(epochs), batch_size=500, verbose=False, cuda=False)
    generator.fit(pd.DataFrame(X, columns=columns))
    np.save(path_out, generator.sample(int(n_needed)).to_numpy(dtype=np.float32))


if __name__ == "__main__":
    main()
