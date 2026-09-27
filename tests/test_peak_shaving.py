from datetime import datetime, timedelta, timezone
import pytest

from terminalweave.planning.peak_shaving import (
    SiteLoad, LoadKind, LoadPowerState, PriorityCalculator,
    SitePeakShaver, PeakShavingProgram,
)
from terminalweave.core.errors import (
    ValidationError, SafetyConstraintError, StateTransitionError, NotFoundError,
)

UTC = timezone.utc
D = datetime(2026, 9, 27, 10, tzinfo=UTC)


def reefer(cid, draw, risk=0.0, protected=False):
    return SiteLoad(cid, LoadKind.REEFER, draw, protected=protected, temp_risk=risk)


def charger(cid, draw, kwh, hours, eff=0.9, start=None):
    return SiteLoad(cid, LoadKind.CHARGING, draw, charge_efficiency=eff,
                    remaining_kwh=kwh, available_from=start,
                    required_by=D + timedelta(hours=hours))


def test_reefer_and_charging_compete_for_one_capacity_constraint():
    shaver = SitePeakShaver(capacity_kw=10.0)
    loads = [reefer('R1', 6.0, risk=0.9), charger('C1', 5.0, 20.0, 8)]
    plan = shaver.plan(loads, D)
    assert plan.served == ('R1',)
    assert [d.load_id for d in plan.deferred] == ['C1']
    assert plan.served_kw == pytest.approx(6.0)
    assert plan.deferred_kw == pytest.approx(5.0)


def test_priority_blends_risk_urgency_and_efficiency():
    calc = PriorityCalculator()
    cold = calc.factors(reefer('R', 5.0, risk=0.8), D)
    tight = calc.factors(charger('C', 5.0, 20.0, 4), D)  # need ~4.4h
    slack = calc.factors(charger('C', 5.0, 20.0, 40), D)
    assert cold.urgency is None and cold.efficiency is None
    assert cold.total == pytest.approx(0.8)
    assert tight.urgency > slack.urgency
    assert tight.total > slack.total
    # a high-efficiency charger scores above a low-efficiency one, all else equal
    fast = calc.factors(charger('F', 5.0, 20.0, 40, eff=0.98), D)
    slow = calc.factors(charger('S', 5.0, 20.0, 40, eff=0.80), D)
    assert fast.total > slow.total


def test_infeasible_deadline_is_flagged():
    factors = PriorityCalculator().factors(charger('C', 4.0, 30.0, 1, eff=0.9), D)
    assert factors.infeasible and factors.urgency == 1.0


def test_risk_can_outrank_a_charging_deadline():
    shaver = SitePeakShaver(capacity_kw=5.0)
    plan = shaver.plan([reefer('R', 5.0, risk=0.95),
                        charger('C', 5.0, 20.0, 40, eff=0.8)], D)
    assert plan.served == ('R',)
    assert plan.deferred[0].load_id == 'C'


def test_protected_load_is_never_shed_and_fails_closed():
    shaver = SitePeakShaver(capacity_kw=4.0)
    with pytest.raises(SafetyConstraintError):
        shaver.plan([reefer('P', 5.0, protected=True)], D)


def test_protected_load_served_ahead_of_higher_scored_unprotected():
    shaver = SitePeakShaver(capacity_kw=6.0)
    plan = shaver.plan([charger('C', 5.0, 5.0, 2, eff=0.99),
                        reefer('P', 5.0, risk=0.0, protected=True)], D)
    assert plan.served == ('P',)
    assert plan.deferred[0].load_id == 'C'


def test_deferral_decision_explains_every_factor():
    plan = SitePeakShaver(capacity_kw=4.0).plan(
        [charger('C', 5.0, 20.0, 40, eff=0.83)], D)
    reason = plan.deferred[0].reason
    assert 'thermal risk 0.00' in reason
    assert 'deadline urgency' in reason
    assert 'charging efficiency 0.83' in reason
    assert 'headroom 4.00 kW below 5.00 kW draw' in reason


def test_reserve_fraction_reduces_usable_capacity():
    shaver = SitePeakShaver(capacity_kw=10.0, reserve_fraction=0.2)
    assert shaver.usable_kw == pytest.approx(8.0)
    plan = shaver.plan([reefer('R', 8.5)], D)
    assert plan.served == () and plan.deferred[0].load_id == 'R'


def test_load_validation_by_kind():
    with pytest.raises(ValidationError):
        SiteLoad('R', LoadKind.REEFER, 5.0, remaining_kwh=10.0)
    with pytest.raises(ValidationError):
        SiteLoad('C', LoadKind.CHARGING, 5.0)
    with pytest.raises(ValidationError):
        charger('C', 5.0, 20.0, hours=-1)
    with pytest.raises(ValidationError):
        PriorityCalculator(risk_weight=0.5, urgency_weight=0.1, efficiency_weight=0.1)


def test_program_sheds_and_then_restores_only_unfinished_demand():
    prog = PeakShavingProgram(capacity_kw=9.0)
    r = reefer('R', 5.0, risk=0.95, protected=True)
    c1 = charger('C1', 4.0, 20.0, 10, eff=0.9)
    c2 = charger('C2', 4.0, 20.0, 40, eff=0.85)
    for load in (r, c1, c2):
        prog.register(load)

    first = prog.evaluate(D)
    assert first.served == ('R', 'C1')
    assert prog.state('C2') == LoadPowerState.DEFERRED
    assert prog.backlog()[0].load_id == 'C2'

    # completed charging demand must never be re-queued
    prog.report_progress('C1', 20.0, D + timedelta(hours=2))
    assert prog.state('C1') == LoadPowerState.DONE
    tight = prog.evaluate(D + timedelta(hours=2), capacity_kw=5.0)
    assert tight.served == ('R',)
    assert 'C1' not in tight.served and [d.load_id for d in tight.deferred] == ['C2']

    # capacity recovers: only the unfinished C2 comes back
    recovered = prog.evaluate(D + timedelta(hours=3), capacity_kw=20.0)
    assert recovered.served == ('R', 'C2')
    assert prog.last_restored == ('C2',)
    assert prog.state('C2') == LoadPowerState.ACTIVE
    assert prog.deferral_minutes('C2', D + timedelta(hours=3)) == pytest.approx(180.0)


def test_deferral_history_accumulates_and_carries_unfinished_energy():
    prog = PeakShavingProgram(capacity_kw=5.0)
    prog.register(reefer('R', 5.0, risk=0.9))
    prog.register(charger('C', 4.0, 20.0, 10))
    assert prog.evaluate(D).served == ('R',)
    prog.evaluate(D + timedelta(hours=1))
    history = prog.deferral_history('C')
    assert len(history) == 2
    assert 'cumulative deferred time 60.0 min' in history[1].reason
    assert 'unfinished demand 20.00 kWh' in history[1].reason


def test_partial_progress_keeps_unfinished_charge_in_backlog():
    prog = PeakShavingProgram(capacity_kw=5.0)
    prog.register(charger('C', 4.0, 20.0, 10))
    prog.evaluate(D)
    prog.report_progress('C', 12.0, D + timedelta(hours=1))
    assert prog.state('C') == LoadPowerState.ACTIVE
    prog.evaluate(D + timedelta(hours=1), capacity_kw=3.0)
    assert prog.state('C') == LoadPowerState.DEFERRED
    restored = prog.evaluate(D + timedelta(hours=2), capacity_kw=10.0)
    assert restored.served == ('C',) and prog.last_restored == ('C',)
    decision = prog.deferral_history('C')[-1]
    assert 'unfinished demand 8.00 kWh' in decision.reason


def test_progress_rejected_while_deferred_or_done():
    prog = PeakShavingProgram(capacity_kw=5.0)
    prog.register(reefer('R', 5.0, risk=0.9))
    prog.register(charger('C', 4.0, 20.0, 10))
    prog.evaluate(D)
    with pytest.raises(StateTransitionError):
        prog.report_progress('C', 1.0, D)
    with pytest.raises(ValidationError):
        prog.report_progress('R', 1.0, D)
    with pytest.raises(NotFoundError):
        prog.state('missing')


def test_duplicate_registration_and_duplicate_plan_ids_rejected():
    prog = PeakShavingProgram(capacity_kw=10.0)
    prog.register(reefer('R', 5.0))
    with pytest.raises(ValidationError):
        prog.register(reefer('R', 5.0))
    with pytest.raises(ValidationError):
        SitePeakShaver(10.0).plan([reefer('R', 5.0), reefer('R', 4.0)], D)
