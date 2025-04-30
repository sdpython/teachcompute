import time
import cProfile
import pstats
import io
from pstats import SortKey
import numpy as np
import pandas
import mmat


def b1(repeat, m1, m2, warmup=3):
    begin = time.time()
    for _ in range(repeat):
        if _ == warmup:
            begin = time.time()
        mmat.mmat(m1, m2, block_size=16, version=1)
    return time.time() - begin


def b2(repeat, m1, m2, warmup=3):
    begin = time.time()
    for _ in range(repeat):
        if _ == warmup:
            begin = time.time()
        m1 @ m2
    return time.time() - begin


def benchmark(run_profile=False):
    repeat = 20
    dtype = np.float32
    data = []
    dims = [2**e for e in range(4, 12)]
    dims = [64, 128, 256, 400, 500, 512, 1000, 1024, 1500, 1800]
    for dim in dims:
        print(f"dim={dim}, repeat={repeat}...")
        m1 = np.random.rand(dim, dim).astype(dtype)
        m2 = np.random.rand(dim, dim).astype(dtype)

        # on vérifie que le code est correct
        if dim < 128:
            res = mmat.mmat(m1, m2)
            np.testing.assert_allclose(m1 @ m2, res, atol=1e-4)

        # benchmark
        t1 = b1(repeat, m1, m2)
        t2 = b2(repeat, m1, m2)
        print(f"dim={dim}, t1={t1}, t2={t2}")

        data.append(dict(repeat=repeat, dim=dim, tmmat=t1, tnumpy=t2))
        if run_profile:
            pr = cProfile.Profile()
            pr.enable()
            b2(repeat, m1, m2)
            pr.disable()
            s = io.StringIO()
            sortby = SortKey.CUMULATIVE
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())
    return data


data = benchmark()
df = pandas.DataFrame(data)
df["ratio"] = df.tmmat / df.tnumpy
print(df)
