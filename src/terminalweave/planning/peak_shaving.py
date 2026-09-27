from __future__ import annotations
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from terminalweave.core.clock import ensure_utc
from terminalweave.core.errors import ValidationError, SafetyConstraintError, StateTransitionError, NotFoundError

class LoadKind(str, Enum):
    REEFER = 'reefer'
    CHARGING = 'charging'

class LoadPowerState(str, Enum):
    ACTIVE = 'active'
    DEFERRED = 'deferred'
    DONE = 'done'

@dataclass(frozen=True)
class SiteLoad:
    """A site electrical load competing for the same peak-shaving capacity.

    Reefers are continuous loads (no deadline, no finite energy budget).
    Charging loads carry a task deadline, a remaining battery energy demand
    and a wall-to-battery efficiency.
    """
    load_id: str
    kind: LoadKind
    draw_kw: float
    protected: bool = False
    temp_risk: float = 0.0
    required_by: datetime | None = None
    available_from: datetime | None = None
    charge_efficiency: float = 1.0
    remaining_kwh: float | None = None

    def __post_init__(self):
        if not self.load_id or not self.load_id.strip():
            raise ValidationError('load id required')
        if not isinstance(self.kind, LoadKind):
            raise ValidationError('invalid load kind')
        if self.draw_kw <= 0:
            raise ValidationError('draw_kw must be positive')
        if not 0 <= self.temp_risk <= 1:
            raise ValidationError('temp_risk must be 0..1')
        if not 0 < self.charge_efficiency <= 1:
            raise ValidationError('charge_efficiency must be 0 < e <= 1')
        if self.kind == LoadKind.REEFER:
            if self.required_by is not None or self.remaining_kwh is not None:
                raise ValidationError('reefer loads are continuous and carry no deadline or energy budget')
        else:
            if self.required_by is None or self.remaining_kwh is None:
                raise ValidationError('charging loads require required_by and remaining_kwh')
            if self.remaining_kwh < 0:
                raise ValidationError('remaining_kwh cannot be negative')
            object.__setattr__(self, 'required_by', ensure_utc(self.required_by))
            if self.available_from is not None:
                object.__setattr__(self, 'available_from', ensure_utc(self.available_from))
                if self.required_by <= self.available_from:
                    raise ValidationError('required_by must be after available_from')

@dataclass(frozen=True)
class PriorityFactors:
    risk: float
    urgency: float | None
    efficiency: float | None
    infeasible: bool
    total: float

@dataclass(frozen=True)
class DeferralDecision:
    load_id: str
    kind: LoadKind
    draw_kw: float
    headroom_kw: float
    factors: PriorityFactors
    reason: str

@dataclass(frozen=True)
class PeakShavingPlan:
    at: datetime
    capacity_kw: float
    served: tuple[str, ...]
    deferred: tuple[DeferralDecision, ...]
    served_kw: float
    deferred_kw: float

class PriorityCalculator:
    """Scores loads from thermal risk, deadline urgency and charging efficiency.

    Only dimensions that apply to a load enter its score; the weights are
    renormalized over the applicable dimensions so a continuous reefer is not
    systematically ranked below deadline-driven charging loads.
    """

    def __init__(self, risk_weight: float=0.45, urgency_weight: float=0.45, efficiency_weight: float=0.10):
        weights = (risk_weight, urgency_weight, efficiency_weight)
        if any(w < 0 for w in weights) or abs(sum(weights) - 1.0) > 1e-9:
            raise ValidationError('priority weights must be non-negative and sum to 1.0')
        self.risk_weight = risk_weight
        self.urgency_weight = urgency_weight
        self.efficiency_weight = efficiency_weight

    def _urgency(self, load: SiteLoad, now: datetime) -> tuple[float, bool]:
        if load.required_by is None:
            return (0.0, False)
        earliest = max(now, load.available_from) if load.available_from is not None else now
        slack_s = (load.required_by - earliest).total_seconds()
        needed_h = (load.remaining_kwh or 0.0) / (load.draw_kw * load.charge_efficiency)
        needed_s = needed_h * 3600
        infeasible = needed_s > slack_s
        if slack_s <= 0:
            urgency = 1.0
        else:
            urgency = min(1.0, needed_s / slack_s)
        return (urgency, infeasible)

    def factors(self, load: SiteLoad, now: datetime) -> PriorityFactors:
        now = ensure_utc(now)
        urgency, infeasible = self._urgency(load, now)
        efficiency = load.charge_efficiency if load.kind == LoadKind.CHARGING else None
        terms: list[tuple[float, float]] = []
        if load.kind == LoadKind.REEFER or load.temp_risk > 0:
            terms.append((self.risk_weight, load.temp_risk))
        if load.required_by is not None:
            terms.append((self.urgency_weight, urgency))
        if efficiency is not None:
            terms.append((self.efficiency_weight, efficiency))
        weight_sum = sum(w for w, _ in terms)
        total = sum(w * v for w, v in terms) / weight_sum if weight_sum else 0.0
        return PriorityFactors(load.temp_risk, urgency if load.required_by is not None else None,
                               efficiency, infeasible, total)

class SitePeakShaver:
    """Single-site peak-shaving pass over one shared capacity constraint."""

    def __init__(self, capacity_kw: float, reserve_fraction: float=0.0,
                 calculator: PriorityCalculator | None=None):
        if capacity_kw <= 0 or not 0 <= reserve_fraction < 1:
            raise ValidationError('invalid site capacity')
        self.capacity_kw = capacity_kw
        self.reserve_fraction = reserve_fraction
        self.usable_kw = capacity_kw * (1 - reserve_fraction)
        self.calculator = calculator or PriorityCalculator()

    def _rank(self, loads: list[SiteLoad], now: datetime) -> list[tuple[PriorityFactors, SiteLoad]]:
        scored = [(self.calculator.factors(load, now), load) for load in loads]
        scored.sort(key=lambda x: (not x[1].protected, -x[0].total, -x[1].draw_kw, x[1].load_id))
        return scored

    def _reason(self, load: SiteLoad, f: PriorityFactors, headroom_kw: float) -> str:
        parts = [f"priority {f.total:.3f} from thermal risk {f.risk:.2f}"]
        if f.urgency is not None:
            parts.append(f"deadline urgency {f.urgency:.2f}")
        if f.efficiency is not None:
            parts.append(f"charging efficiency {f.efficiency:.2f}")
        text = '; '.join(parts)
        if f.infeasible:
            text += '; deadline already infeasible at full draw'
        text += f'; site headroom {headroom_kw:.2f} kW below {load.draw_kw:.2f} kW draw'
        return text

    def plan(self, loads: list[SiteLoad], now: datetime) -> PeakShavingPlan:
        now = ensure_utc(now)
        ids = [load.load_id for load in loads]
        if len(set(ids)) != len(ids):
            raise ValidationError('duplicate load id in plan request')
        served: list[str] = []
        deferred: list[DeferralDecision] = []
        used_kw = 0.0
        for f, load in self._rank(loads, now):
            if used_kw + load.draw_kw <= self.usable_kw + 1e-9:
                served.append(load.load_id)
                used_kw += load.draw_kw
            elif load.protected:
                raise SafetyConstraintError(
                    f'protected load {load.load_id} cannot be shed and exceeds site capacity')
            else:
                headroom = max(0.0, self.usable_kw - used_kw)
                deferred.append(DeferralDecision(
                    load.load_id, load.kind, load.draw_kw, headroom, f,
                    self._reason(load, f, headroom)))
        deferred.sort(key=lambda d: d.load_id)
        return PeakShavingPlan(now, self.capacity_kw, tuple(served), tuple(deferred),
                               used_kw, sum(d.draw_kw for d in deferred))

@dataclass
class _Track:
    load: SiteLoad
    state: LoadPowerState = LoadPowerState.ACTIVE
    deferred_since: datetime | None = None
    deferred_seconds: float = 0.0
    history: list[DeferralDecision] = field(default_factory=list)

class PeakShavingProgram:
    """Stateful site peak-shaving program across capacity changes.

    Deferred loads keep their place as unfinished demand and are reconsidered
    whenever capacity recovers; completed demands are never re-queued. Every
    deferral is recorded with an explicit, cumulative explanation.
    """

    def __init__(self, capacity_kw: float, reserve_fraction: float=0.0,
                 calculator: PriorityCalculator | None=None):
        self.capacity_kw = capacity_kw
        self.reserve_fraction = reserve_fraction
        self.calculator = calculator or PriorityCalculator()
        self._tracks: dict[str, _Track] = {}
        self.last_plan: PeakShavingPlan | None = None
        self.last_restored: tuple[str, ...] = ()

    def register(self, load: SiteLoad) -> None:
        track = self._tracks.get(load.load_id)
        if track is not None and track.state != LoadPowerState.DONE:
            raise ValidationError(f'load {load.load_id} already registered')
        self._tracks[load.load_id] = _Track(load)

    def remove(self, load_id: str) -> None:
        if load_id not in self._tracks:
            raise NotFoundError(f'unknown load {load_id}')
        del self._tracks[load_id]

    def state(self, load_id: str) -> LoadPowerState:
        return self._track(load_id).state

    def _track(self, load_id: str) -> _Track:
        try:
            return self._tracks[load_id]
        except KeyError as exc:
            raise NotFoundError(f'unknown load {load_id}') from exc

    def backlog(self) -> list[SiteLoad]:
        return [t.load for t in self._tracks.values() if t.state == LoadPowerState.DEFERRED]

    def deferral_minutes(self, load_id: str, now: datetime | None=None) -> float:
        t = self._track(load_id)
        seconds = t.deferred_seconds
        if t.deferred_since is not None:
            end = ensure_utc(now) if now is not None else t.deferred_since
            seconds += (end - t.deferred_since).total_seconds()
        return max(0.0, seconds) / 60

    def deferral_history(self, load_id: str) -> tuple[DeferralDecision, ...]:
        return tuple(self._track(load_id).history)

    def complete(self, load_id: str) -> None:
        t = self._track(load_id)
        if t.state == LoadPowerState.DONE:
            return
        t.state = LoadPowerState.DONE
        t.deferred_since = None

    def report_progress(self, load_id: str, delivered_kwh: float, now: datetime) -> SiteLoad:
        t = self._track(load_id)
        if t.state == LoadPowerState.DONE:
            raise StateTransitionError(f'load {load_id} already completed')
        if t.state == LoadPowerState.DEFERRED:
            raise StateTransitionError(f'load {load_id} is shed and cannot receive charge')
        if t.load.kind != LoadKind.CHARGING or t.load.remaining_kwh is None:
            raise ValidationError('only charging loads report energy progress')
        if delivered_kwh <= 0 or delivered_kwh > t.load.remaining_kwh + 1e-9:
            raise ValidationError('delivered energy outside remaining demand')
        remaining = t.load.remaining_kwh - delivered_kwh
        t.load = replace(t.load, remaining_kwh=remaining)
        if remaining <= 1e-9:
            t.state = LoadPowerState.DONE
        return t.load

    def _enrich(self, decision: DeferralDecision, track: _Track, now: datetime) -> DeferralDecision:
        minutes = track.deferred_seconds / 60
        if track.deferred_since is not None:
            minutes += (now - track.deferred_since).total_seconds() / 60
        reason = f'{decision.reason}; cumulative deferred time {minutes:.1f} min'
        unfinished = track.load.remaining_kwh
        if unfinished is not None:
            reason += f'; unfinished demand {unfinished:.2f} kWh'
        return replace(decision, reason=reason)

    def evaluate(self, now: datetime, capacity_kw: float | None=None) -> PeakShavingPlan:
        now = ensure_utc(now)
        capacity = self.capacity_kw if capacity_kw is None else capacity_kw
        shaver = SitePeakShaver(capacity, self.reserve_fraction, self.calculator)
        active = [t.load for t in self._tracks.values() if t.state != LoadPowerState.DONE]
        plan = shaver.plan(active, now)
        served = set(plan.served)
        restored: list[str] = []
        for track in self._tracks.values():
            if track.state == LoadPowerState.DONE:
                continue
            if track.load.load_id in served:
                if track.state == LoadPowerState.DEFERRED and track.deferred_since is not None:
                    track.deferred_seconds += (now - track.deferred_since).total_seconds()
                    track.deferred_since = None
                    restored.append(track.load.load_id)
                track.state = LoadPowerState.ACTIVE
            else:
                if track.state != LoadPowerState.DEFERRED:
                    track.deferred_since = now
                track.state = LoadPowerState.DEFERRED
        for decision in plan.deferred:
            track = self._tracks[decision.load_id]
            enriched = self._enrich(decision, track, now)
            track.history.append(enriched)
        restored.sort()
        self.last_plan = plan
        self.last_restored = tuple(restored)
        return plan
