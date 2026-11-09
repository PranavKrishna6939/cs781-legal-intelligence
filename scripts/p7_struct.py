import sys

import numpy as np

sys.path.insert(0, "scripts")
from p7_bm25 import K, write_zip
from p7_fix import FUSE, scored
from p7_probe2 import pick
from p7_probe3 import col_softmax


def main():
    tq, qids, cand, views, self_mask, after = scored("Test")
    base = sum(FUSE[k] * views[k] for k in FUSE)
    base[self_mask] = -50
    base[after] = -50
    subs = {
        "P7_struct_date": pick(qids, cand, base),
        "P7_struct_col05": pick(qids, cand, col_softmax(base, 0.5)),
        "P7_struct_col02": pick(qids, cand, col_softmax(base, 0.2)),
    }
    for name, pred in subs.items():
        assert len(pred) == len(tq) and all(len(set(v)) == K and q not in v for q, v in pred.items())
        write_zip(name, pred)
        print("wrote", name)


if __name__ == "__main__":
    main()
