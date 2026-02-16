from prism_ssl.model.schedules import supcon_weight


def test_supcon_schedule_warmup_and_ramp():
    assert supcon_weight(step=0, warmup=10, ramp=20, target=0.2) == 0.0
    assert supcon_weight(step=9, warmup=10, ramp=20, target=0.2) == 0.0
    assert supcon_weight(step=10, warmup=10, ramp=20, target=0.2) == 0.0
    assert supcon_weight(step=20, warmup=10, ramp=20, target=0.2) == 0.1
    assert supcon_weight(step=30, warmup=10, ramp=20, target=0.2) == 0.2
    assert supcon_weight(step=100, warmup=10, ramp=20, target=0.2) == 0.2


def test_supcon_schedule_zero_ramp():
    assert supcon_weight(step=11, warmup=10, ramp=0, target=0.5) == 0.5
