import copy

import pytest

from pdm.web_collect import TASKS, verify_state


def test_ride_receipt_requires_requested_route_in_order_class_payment_and_completion():
    task = TASKS[2]
    receipt = {
        "status": "completed", "pickup": {"name": task.pickup},
        "destination": {"name": task.destination}, "car": {"type": task.car},
        "paymentMethod": {"type": "cash"},
    }

    def check(record):
        return verify_state(task, {"finalstate": {"trips": {"pastTrips": [record]}}}, {})

    assert check(receipt) == receipt
    for change in (
        {"pickup": receipt["destination"], "destination": receipt["pickup"]},
        {"paymentMethod": {"type": "credit"}}, {"car": {"type": "Comfort"}},
        {"status": "pending"},
    ):
        with pytest.raises(ValueError, match="No matching completed booking"):
            check({**receipt, **change})


def test_reservation_requires_matching_party_time_and_contact():
    task = TASKS[0]
    chosen = {"restaurant": "Vintage Bites", "time": "12:30 PM"}
    details = {
        "restaurant": {"name": chosen["restaurant"]}, "time": chosen["time"],
        "date": "2026-11-24", "guests": "2 people", "tel": "+1 415 555 0146",
    }
    state = {"finalstate": {"booking": {"bookingDetails": [details]}}}
    assert verify_state(task, state, chosen)["date"] == "2026-11-24"
    for change in ({"guests": "3 people"}, {"time": "11:00 AM"}, {"tel": "+1 415 555 0147"}):
        wrong = copy.deepcopy(state)
        wrong["finalstate"]["booking"]["bookingDetails"][0].update(change)
        with pytest.raises(ValueError, match="No matching completed booking"):
            verify_state(task, wrong, chosen)
