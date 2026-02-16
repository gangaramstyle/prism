# Test Scaffold

Implemented coverage:
1. `test_hashing.py` for deterministic shard assignment.
2. `test_sharded_dataset.py` for warm-pool bootstrap, replacement lifecycle, and broken-ratio abort behavior.
3. `test_loss_schedule.py` for SupCon ramp behavior.
4. `test_supcon.py` for supervised contrastive edge cases.
5. `test_train_smoke.py` for 120-step CPU integration and resume checkpoint selection priority.
