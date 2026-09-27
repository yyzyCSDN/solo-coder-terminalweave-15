from datetime import datetime, timedelta, timezone
import pytest

from terminalweave.core.errors import SafetyConstraintError, ValidationError
from terminalweave.planning.peak_shaving import (
    CapacityWindow,
    ChargeDemand,
    DemandKind,
    PeakShavePlanner,
    PeakShavePolicy,
    ReeferDemand,
)

UTC = timezone.utc
D = datetime(2026, 1, 1, tzinfo=UTC)

def win(hours_from_start, kw, length_hours=1):
    start = D + timedelta(hours=hours_from_start)
    return CapacityWindow(start, start + timedelta(hours=length_hours), kw)

def reefer(cid, temp, drift=2.0, kw=4.0, protected=False, band=(-25.0, -18.0)):
    return ReeferDemand(cid, kw, temp, band[0], band[1], drift, protected)

def charge(eid, kwh, required_by, eff=1.0, kw=8.0, protected=False, available=None):
    return ChargeDemand(eid, kw, kwh, eff, available or D, required_by, protected)

def test_shared_capacity_serves_reefer_and_charging():
    plan = PeakShavePlanner().plan([win(0, 100)], [reefer("C1", -21.5)], [charge("E1", 8, D + timedelta(hours=4))])
    assert {(s.kind, s.demand_id) for s in plan.scheduled} == {(DemandKind.REEFER, "C1"), (DemandKind.CHARGING, "E1")}
    assert sum(s.allocated_kw for s in plan.scheduled) <= 100
    assert plan.deferred == () and plan.unmet == ()

def test_temperature_risk_outranks_safe_reefer():
    risky = reefer("RISKY", -18.3)   # 0.15h from the -18C edge
    safe = reefer("SAFE", -21.5)     # mid-band, 1.75h from either edge
    plan = PeakShavePlanner().plan([win(0, 4)], [safe, risky], [])
    assert [s.demand_id for s in plan.scheduled] == ["RISKY"]
    (deferral,) = plan.deferred
    assert deferral.demand_id == "SAFE"
    assert deferral.outranked_by == ("reefer:RISKY",)
    assert "reefer:RISKY" in deferral.reason and "band edge" in deferral.reason

def test_charge_deadline_sets_priority():
    tight = charge("TIGHT", 8, D + timedelta(hours=2))
    loose = charge("LOOSE", 8, D + timedelta(hours=10))
    plan = PeakShavePlanner().plan([win(0, 8)], [], [loose, tight])
    assert [s.demand_id for s in plan.scheduled] == ["TIGHT"]
    assert plan.deferred[0].demand_id == "LOOSE"

def test_charge_efficiency_breaks_deadline_tie():
    efficient = charge("EFF", 8, D + timedelta(hours=10), eff=0.96)
    lossy = charge("LOSSY", 8, D + timedelta(hours=10), eff=0.5)
    plan = PeakShavePlanner().plan([win(0, 8)], [], [lossy, efficient])
    assert [s.demand_id for s in plan.scheduled] == ["EFF"]
    deferral = plan.deferred[0]
    assert deferral.demand_id == "LOSSY" and "efficiency" in deferral.reason

def test_protected_load_is_never_shed():
    guarded = reefer("GUARDED", -21.5, drift=0.0, protected=True)  # zero urgency
    urgent = charge("URGENT", 8, D + timedelta(hours=1))           # maximal urgency
    plan = PeakShavePlanner().plan([win(0, 4)], [guarded], [urgent])
    assert [s.demand_id for s in plan.scheduled] == ["GUARDED"]
    assert plan.deferred[0].demand_id == "URGENT"
    assert plan.deferred[0].outranked_by == ("reefer:GUARDED",)

def test_protected_load_beyond_capacity_fails_closed():
    guarded = reefer("GUARDED", -21.5, kw=10.0, protected=True)
    with pytest.raises(SafetyConstraintError):
        PeakShavePlanner().plan([win(0, 5)], [guarded], [])

def test_recovery_reschedules_only_unmet_demand():
    windows = [win(0, 8), win(1, 100), win(2, 100), win(3, 100)]
    first = charge("FIRST", 8, D + timedelta(hours=1))    # wins the peak window
    backlog = charge("BACKLOG", 16, D + timedelta(hours=10))  # needs 2h of charge
    plan = PeakShavePlanner().plan(windows, [], [backlog, first])
    assert [s.window_start for s in plan.scheduled_for("FIRST")] == [D]
    assert [s.window_start for s in plan.scheduled_for("BACKLOG")] == [D + timedelta(hours=1), D + timedelta(hours=2)]
    (deferral,) = plan.deferrals_for("BACKLOG")
    assert deferral.window_start == D
    assert "outranked by [charging:FIRST]" in deferral.reason
    assert "16.0kWh still owed" in deferral.reason
    assert len(plan.scheduled) == 3 and plan.unmet == ()

def test_unmet_demand_listed_when_horizon_too_short():
    plan = PeakShavePlanner().plan([win(0, 4)], [reefer("C1", -17.0, drift=0.0)], [charge("E1", 16, D + timedelta(hours=10), kw=4.0)])
    assert [s.demand_id for s in plan.scheduled] == ["C1"]  # out-of-band reefer wins the window
    assert plan.deferred[0].demand_id == "E1"
    assert "16.0kWh still owed" in plan.deferred[0].reason
    assert plan.unmet == ("charging:E1", "reefer:C1")

def test_shed_reefer_gains_urgency_as_temperature_drifts():
    windows = [win(0, 4), win(1, 4)]
    plan = PeakShavePlanner().plan(windows, [reefer("A", -21.5), reefer("B", -21.5)], [])
    assert [s.demand_id for s in plan.scheduled if s.window_start == D] == ["A"]
    assert [s.demand_id for s in plan.scheduled if s.window_start == D + timedelta(hours=1)] == ["B"]
    assert plan.deferrals_for("B")[0].window_start == D

def test_plan_is_deterministic():
    windows = [win(0, 8), win(1, 100)]
    reefers = [reefer("C1", -19.0), reefer("C2", -21.5)]
    chargers = [charge("E1", 8, D + timedelta(hours=3), eff=0.9), charge("E2", 8, D + timedelta(hours=6), eff=0.8)]
    assert PeakShavePlanner().plan(windows, reefers, chargers) == PeakShavePlanner(PeakShavePolicy()).plan(windows, reefers, chargers)

def test_validation_rejects_bad_input():
    with pytest.raises(ValidationError):
        CapacityWindow(datetime(2026, 1, 1), D + timedelta(hours=1), 10)  # naive datetime
    with pytest.raises(ValidationError):
        win(0, -1)
    with pytest.raises(ValidationError):
        charge("E1", 8, D + timedelta(hours=1), eff=0.0)
    with pytest.raises(ValidationError):
        reefer("C1", -21.5, band=(-18.0, -25.0))
    with pytest.raises(ValidationError):
        PeakShavePlanner().plan([win(0, 10, 2), win(1, 10)], [], [])
    with pytest.raises(ValidationError):
        PeakShavePlanner().plan([win(0, 10)], [reefer("C1", -21.5), reefer("C1", -20.0)], [])
