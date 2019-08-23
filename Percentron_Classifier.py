
import sys
import numpy as np
import pandas as pd

max_it = 30000

def perceptron(df):
    mat = df.values.astype(np.float)
    (n, d) = mat.shape
    x, y = mat[:, 0: d-1], mat[:, d-1]
    x = np.column_stack((x, np.ones(n)))
    db = []
    w = np.zeros(d)

    for t in range(max_it):
        db.append(w)
        converged = True
        for i in range(n):
            if x[i].dot(w) * y[i] <= 0:
                w = w + x[i] * y[i]
                converged = False
        if converged:
            break
    db = pd.DataFrame(db)
    return db

def main():
    file_in = sys.argv[1]
    file_ot = sys.argv[2]
    dsn = pd.read_csv(file_in, header=None)
    result = perceptron(dsn)
    result.to_csv(file_ot, header=False, index=False)

if __name__ == '__main__':
    main()