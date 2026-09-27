"""Facility-wide peak-shaving plan shared by reefer loads and EV charging.

Reefer (refrigerated container) loads and electric-equipment charging draw
from one capacity budget per time window.  When a window cannot serve every
demand, priority is scored from temperature-control risk (reefers) and from
task deadlines plus charging efficiency (equipment).  Protected loads are
never shed: if capacity cannot even cover protected demand the planner fails
closed.  Deferred demand stays in the backlog, so once capacity recovers only
unmet demand is rescheduled, and every deferral carries a reason.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from terminalweave.core.clock import ensure_utc
from terminalweave.core.errors import ValidationError, SafetyConstraintError

_EPS = 1e-09

class DemandKind(str, Enum):
    REEFER = 'reefer'
    CHARGING = 'charging'

@dataclass(frozen=True)
class CapacityWindow:
    start: datetime
    end: datetime
    available_kw: float

    def __post_init__(self):
        object.__setattr__(self, 'start', ensure_utc(self.start))
        object.__setattr__(self, 'end', ensure_utc(self.end))
        if self.end <= self.start:
            raise ValidationError('window end must be after start')
        if self.available_kw < 0:
            raise ValidationError('window capacity cannot be negative')

    @property
    def hours(self) -> float:
        return (self.end - self.start).total_seconds() / 3600

@dataclass(frozen=True)
class ReeferDemand:
    container_id: str
    demand_kw: float
    current_temp_c: float
    low_c: float
    high_c: float
    drift_c_per_hour: float
    protected: bool = False

    def __post_init__(self):
        if self.demand_kw <= 0 or self.drift_c_per_hour < 0:
            raise ValidationError('invalid reefer demand')
        if self.low_c >= self.high_c:
            raise ValidationError('reefer band low must be below high')
        if not -80 <= self.current_temp_c <= 80:
            raise ValidationError('temperature outside supported range')

@dataclass(frozen=True)
class ChargeDemand:
    equipment_id: str
    demand_kw: float
    energy_kwh: float
    efficiency: float
    available_from: datetime
    required_by: datetime
    protected: bool = False

    def __post_init__(self):
        object.__setattr__(self, 'available_from', ensure_utc(self.available_from))
        object.__setattr__(self, 'required_by', ensure_utc(self.required_by))
        if self.demand_kw <= 0 or self.energy_kwh <= 0 or not 0 < self.efficiency <= 1:
            raise ValidationError('invalid charge demand')
        if self.required_by <= self.available_from:
            raise ValidationError('required_by must be after available_from')

@dataclass(frozen=True)
class PeakShavePolicy:
    risk_horizon: timedelta = timedelta(hours=2)
    efficiency_weight: float = 0.25

    def __post_init__(self):
        if self.risk_horizon.total_seconds() <= 0:
            raise ValidationError('risk horizon must be positive')
        if self.efficiency_weight < 0:
            raise ValidationError('efficiency weight cannot be negative')

@dataclass(frozen=True)
class ScheduledLoad:
    demand_id: str
    kind: DemandKind
    window_start: datetime
    allocated_kw: float
    score: float

@dataclass(frozen=True)
class DeferredDecision:
    demand_id: str
    kind: DemandKind
    window_start: datetime
    score: float
    shortfall_kw: float
    outranked_by: tuple[str, ...]
    reason: str

@dataclass(frozen=True)
class PeakShavePlan:
    scheduled: tuple[ScheduledLoad, ...]
    deferred: tuple[DeferredDecision, ...]
    unmet: tuple[str, ...]

    def scheduled_for(self, demand_id: str) -> tuple[ScheduledLoad, ...]:
        return tuple((x for x in self.scheduled if x.demand_id == demand_id))

    def deferrals_for(self, demand_id: str) -> tuple[DeferredDecision, ...]:
        return tuple((x for x in self.deferred if x.demand_id == demand_id))

@dataclass(frozen=True)
class _Candidate:
    kind: DemandKind
    demand_id: str
    demand_kw: float
    protected: bool
    score: float
    context: str

    @property
    def label(self) -> str:
        return f'{self.kind.value}:{self.demand_id}'

def _defer_reason(cand: _Candidate, left_kw: float, served: list[str]) -> str:
    who = ', '.join(served) if served else 'none'
    return f'needs {cand.demand_kw:.1f}kW but only {left_kw:.1f}kW unallocated; outranked by [{who}]; {cand.context}'

class PeakShavePlanner:

    def __init__(self, policy: PeakShavePolicy | None=None):
        self.policy = policy or PeakShavePolicy()

    def _horizon_hours(self) -> float:
        return self.policy.risk_horizon.total_seconds() / 3600

    def _reefer_urgency(self, temp_c: float, low_c: float, high_c: float, drift_c_per_hour: float) -> float:
        if temp_c < low_c or temp_c > high_c:
            return 1.0
        if drift_c_per_hour <= 0:
            return 0.0
        hours_to_limit = min(temp_c - low_c, high_c - temp_c) / drift_c_per_hour
        return 1.0 - min(hours_to_limit / self._horizon_hours(), 1.0)

    def _charge_urgency(self, energy_kwh: float, demand_kw: float, efficiency: float, required_by: datetime, window_end: datetime) -> float:
        hours_needed = energy_kwh / (demand_kw * efficiency)
        slack_hours = (required_by - window_end).total_seconds() / 3600 - hours_needed
        if slack_hours <= 0:
            return 1.0
        return 1.0 - min(slack_hours / self._horizon_hours(), 1.0)

    @staticmethod
    def _next_temp(temp_c: float, low_c: float, high_c: float, drift_c_per_hour: float, hours: float, powered: bool) -> float:
        mid = (low_c + high_c) / 2
        step = drift_c_per_hour * hours
        if powered:
            return max(mid, temp_c - step) if temp_c > mid else min(mid, temp_c + step)
        return temp_c + step if temp_c >= mid else temp_c - step

    def _reefer_context(self, temp_c: float, r: ReeferDemand) -> str:
        band = f'[{r.low_c:.1f},{r.high_c:.1f}]C'
        if temp_c < r.low_c or temp_c > r.high_c:
            return f'temp {temp_c:.1f}C outside band {band}'
        if r.drift_c_per_hour > 0:
            hours_to_limit = min(temp_c - r.low_c, r.high_c - temp_c) / r.drift_c_per_hour
            return f'temp {temp_c:.1f}C, ~{hours_to_limit:.2f}h to band edge {band}'
        return f'temp {temp_c:.1f}C stable in band {band}'

    def _charge_context(self, energy_kwh: float, c: ChargeDemand, window: CapacityWindow) -> str:
        hours_needed = energy_kwh / (c.demand_kw * c.efficiency)
        slack = (c.required_by - window.end).total_seconds() / 3600 - hours_needed
        return f'{energy_kwh:.1f}kWh still owed by {c.required_by.isoformat()}, slack {slack:.2f}h, efficiency {c.efficiency:.2f}'

    def plan(self, windows: list[CapacityWindow], reefers: list[ReeferDemand], chargers: list[ChargeDemand]) -> PeakShavePlan:
        windows = sorted(windows, key=lambda w: (w.start, w.end))
        for earlier, later in zip(windows, windows[1:]):
            if later.start < earlier.end:
                raise ValidationError('capacity windows must not overlap')
        for kind, ids in (('reefer', [r.container_id for r in reefers]), ('charging', [c.equipment_id for c in chargers])):
            if len(ids) != len(set(ids)):
                raise ValidationError(f'duplicate {kind} demand id')
        reefer_by_id = {r.container_id: r for r in reefers}
        charger_by_id = {c.equipment_id: c for c in chargers}
        temps = {r.container_id: r.current_temp_c for r in reefers}
        owed = {c.equipment_id: c.energy_kwh for c in chargers}
        scheduled: list[ScheduledLoad] = []
        deferred: list[DeferredDecision] = []
        for window in windows:
            candidates = []
            for r in reefers:
                temp = temps[r.container_id]
                score = self._reefer_urgency(temp, r.low_c, r.high_c, r.drift_c_per_hour)
                candidates.append(_Candidate(DemandKind.REEFER, r.container_id, r.demand_kw, r.protected, score, self._reefer_context(temp, r)))
            for c in chargers:
                if owed[c.equipment_id] <= _EPS or window.start < c.available_from:
                    continue
                score = self._charge_urgency(owed[c.equipment_id], c.demand_kw, c.efficiency, c.required_by, window.end) + self.policy.efficiency_weight * c.efficiency
                candidates.append(_Candidate(DemandKind.CHARGING, c.equipment_id, c.demand_kw, c.protected, score, self._charge_context(owed[c.equipment_id], c, window)))
            candidates.sort(key=lambda x: (not x.protected, -x.score, x.kind.value, x.demand_id))
            left_kw = window.available_kw
            served: list[str] = []
            for cand in candidates:
                if cand.demand_kw <= left_kw + _EPS:
                    left_kw -= cand.demand_kw
                    served.append(cand.label)
                    scheduled.append(ScheduledLoad(cand.demand_id, cand.kind, window.start, cand.demand_kw, cand.score))
                    if cand.kind is DemandKind.REEFER:
                        r = reefer_by_id[cand.demand_id]
                        temps[cand.demand_id] = self._next_temp(temps[cand.demand_id], r.low_c, r.high_c, r.drift_c_per_hour, window.hours, True)
                    else:
                        c = charger_by_id[cand.demand_id]
                        owed[cand.demand_id] = max(0.0, owed[cand.demand_id] - cand.demand_kw * window.hours * c.efficiency)
                else:
                    if cand.protected:
                        raise SafetyConstraintError(f'protected load {cand.label} needs {cand.demand_kw}kW but only {left_kw}kW available in window starting {window.start.isoformat()}')
                    deferred.append(DeferredDecision(cand.demand_id, cand.kind, window.start, cand.score, cand.demand_kw - left_kw, tuple(served), _defer_reason(cand, left_kw, served)))
                    if cand.kind is DemandKind.REEFER:
                        r = reefer_by_id[cand.demand_id]
                        temps[cand.demand_id] = self._next_temp(temps[cand.demand_id], r.low_c, r.high_c, r.drift_c_per_hour, window.hours, False)
        unmet = [f'charging:{eid}' for eid, kwh in owed.items() if kwh > _EPS]
        unmet += [f'reefer:{r.container_id}' for r in reefers if temps[r.container_id] < r.low_c or temps[r.container_id] > r.high_c]
        return PeakShavePlan(tuple(scheduled), tuple(deferred), tuple(sorted(unmet)))
