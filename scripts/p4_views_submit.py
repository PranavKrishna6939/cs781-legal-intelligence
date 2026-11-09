"""Write P4_dense_views_noleak: mean log-odds of the C1 model read at a 2,048-token tail and at a 4,096-token tail."""
from saul.cjpe.ensemble import _align, mix_logits
from saul.cjpe.submit import check_test_hash, predictions_from_logits, test_ids, write_zip


def main() -> None:
    check_test_hash(stop=True)
    ids = test_ids()
    logits = mix_logits([_align("C1", "test", ids, False), _align("C1_t4096", "test", ids, False)])
    write_zip(predictions_from_logits(ids, logits, 0.0), "submissions/phase4/P4_dense_views_noleak", ids)


if __name__ == "__main__":
    main()
