from prism_ssl.utils.hashing import shard_worker, stable_int_hash


def test_stable_int_hash_is_deterministic():
    value = "scan_abc"
    assert stable_int_hash(value) == stable_int_hash(value)


def test_shard_worker_deterministic_and_in_range():
    scan_id = "scan_123"
    shards = [shard_worker(scan_id, 8) for _ in range(10)]
    assert len(set(shards)) == 1
    assert 0 <= shards[0] < 8
