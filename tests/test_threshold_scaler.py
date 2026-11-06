from src.baselines.threshold_scaler import ThresholdScaler


def create_info(
    cpu=50.0,
    memory=50.0,
    latency=100.0,
):
    return {
        "cpu_utilization": cpu,
        "memory_utilization": memory,
        "latency": latency,
    }


def test_threshold_scaler_maintains_under_normal_load():
    scaler = ThresholdScaler(
        cooldown_steps=0,
    )

    scaler.reset()

    action = scaler.select_action(
        create_info(
            cpu=50.0,
            memory=50.0,
            latency=120.0,
        )
    )

    assert action == ThresholdScaler.MAINTAIN


def test_threshold_scaler_scales_up_for_high_cpu():
    scaler = ThresholdScaler(
        cooldown_steps=0,
    )

    scaler.reset()

    action = scaler.select_action(
        create_info(
            cpu=85.0,
            memory=50.0,
            latency=120.0,
        )
    )

    assert action == ThresholdScaler.SCALE_UP


def test_threshold_scaler_scales_up_for_high_memory():
    scaler = ThresholdScaler(
        cooldown_steps=0,
    )

    scaler.reset()

    action = scaler.select_action(
        create_info(
            cpu=50.0,
            memory=85.0,
            latency=120.0,
        )
    )

    assert action == ThresholdScaler.SCALE_UP


def test_threshold_scaler_scales_up_for_high_latency():
    scaler = ThresholdScaler(
        cooldown_steps=0,
    )

    scaler.reset()

    action = scaler.select_action(
        create_info(
            cpu=50.0,
            memory=50.0,
            latency=250.0,
        )
    )

    assert action == ThresholdScaler.SCALE_UP


def test_threshold_scaler_scales_down_for_low_utilization():
    scaler = ThresholdScaler(
        cooldown_steps=0,
    )

    scaler.reset()

    action = scaler.select_action(
        create_info(
            cpu=20.0,
            memory=20.0,
            latency=80.0,
        )
    )

    assert action == ThresholdScaler.SCALE_DOWN


def test_threshold_scaler_does_not_scale_down_if_latency_is_high():
    scaler = ThresholdScaler(
        cooldown_steps=0,
    )

    scaler.reset()

    action = scaler.select_action(
        create_info(
            cpu=20.0,
            memory=20.0,
            latency=150.0,
        )
    )

    assert action == ThresholdScaler.MAINTAIN


def test_threshold_scaler_cooldown():
    scaler = ThresholdScaler(
        cooldown_steps=3,
    )

    scaler.reset()

    first_action = scaler.select_action(
        create_info(
            cpu=90.0,
            memory=90.0,
            latency=300.0,
        )
    )

    second_action = scaler.select_action(
        create_info(
            cpu=90.0,
            memory=90.0,
            latency=300.0,
        )
    )

    third_action = scaler.select_action(
        create_info(
            cpu=90.0,
            memory=90.0,
            latency=300.0,
        )
    )

    fourth_action = scaler.select_action(
        create_info(
            cpu=90.0,
            memory=90.0,
            latency=300.0,
        )
    )

    assert first_action == ThresholdScaler.SCALE_UP
    assert second_action == ThresholdScaler.MAINTAIN
    assert third_action == ThresholdScaler.MAINTAIN
    assert fourth_action == ThresholdScaler.MAINTAIN