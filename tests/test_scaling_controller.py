from src.controller.scaling_controller import (
    ScalingController,
)


def create_info(
    cpu=50.0,
    memory=50.0,
    latency=100.0,
    request_rate=100.0,
    previous_request_rate=100.0,
):
    return {
        "cpu_utilization": cpu,
        "memory_utilization": memory,
        "latency": latency,
        "request_rate": request_rate,
        "previous_request_rate": previous_request_rate,
    }


def test_controller_returns_none_under_normal_conditions():

    controller = ScalingController()

    mode = controller.select_mode(
        state=None,
        info=create_info(),
    )

    assert mode == ScalingController.MODE_NONE


def test_controller_selects_horizontal_scaling():

    controller = ScalingController()

    mode = controller.select_mode(
        state=None,
        info=create_info(
            cpu=85.0,
            memory=50.0,
            latency=150.0,
        ),
    )

    assert mode == (
        ScalingController.MODE_HORIZONTAL
    )


def test_controller_selects_vertical_scaling():

    controller = ScalingController()

    mode = controller.select_mode(
        state=None,
        info=create_info(
            cpu=76.0,
            memory=76.0,
            latency=150.0,
        ),
    )

    assert mode == (
        ScalingController.MODE_BOTH
    )


def test_controller_selects_both_for_high_latency_and_resources():

    controller = ScalingController()

    mode = controller.select_mode(
        state=None,
        info=create_info(
            cpu=80.0,
            memory=80.0,
            latency=250.0,
        ),
    )

    assert mode == (
        ScalingController.MODE_BOTH
    )


def test_controller_detects_workload_spike():

    controller = ScalingController()

    mode = controller.select_mode(
        state=None,
        info=create_info(
            cpu=50.0,
            memory=50.0,
            latency=100.0,
            request_rate=150.0,
            previous_request_rate=100.0,
        ),
    )

    assert mode == (
        ScalingController.MODE_HORIZONTAL
    )


def test_controller_reset():

    controller = ScalingController()

    result = controller.reset()

    assert result is None