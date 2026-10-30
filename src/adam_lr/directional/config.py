"""TOML settings for width-1 product-model experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ModelConfig:
    layers: int = 2
    B: float = 2.0
    Lambda: float = 1.0
    M: float = 35.0


def rho_lower_bound(model: ModelConfig) -> float:
    """Strict initialization threshold rho_0(M,B) from the paper."""
    if (isinstance(model.layers, bool) or not isinstance(model.layers, int)
            or model.layers < 2):
        raise ValueError("model.layers must be an integer >= 2")
    for name in ("B", "M"):
        value = getattr(model, name)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"model.{name} must be finite and positive")
    if model.M <= model.B + 1:
        raise ValueError("model.M must be greater than model.B + 1")
    return max(1.0, (3 * (model.B + 1) * math.log(model.M)
                     / (model.M - model.B - 1)) ** (1 / model.layers))


@dataclass(frozen=True)
class InitializationConfig:
    rho: float = 1.01
    sampling: str = "grid"
    grid_size: int = 11
    sample_count: int = 256
    seed: int = 0
    z_min: float = 0.0
    z_max: float | None = None
    pi_min: float = 0.2
    pi_max: float = 5.0
    z_values: list[float] | None = None
    pi_values: list[float] | None = None
    control_points: list[list[float]] | None = None
    sectors: str = "both"
    max_trajectories: int = 100_000
    require_admissible_rho: bool = True

    def control_bounds(self, model: ModelConfig) -> tuple[float, float, float, float]:
        return self.z_min, 3 * math.log(model.M) if self.z_max is None else self.z_max, self.pi_min, self.pi_max


@dataclass(frozen=True)
class OptimizerConfig:
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 0.0


@dataclass(frozen=True)
class ScheduleConfig:
    initial_rate: float = 0.001
    alpha: float = 0.15


@dataclass(frozen=True)
class RunConfig:
    steps: int = 600_000
    dtype: str = "float64"
    device: str = "cpu"
    threads: int = 1
    output_dir: str = "runs/sphere"


@dataclass(frozen=True)
class LoggingConfig:
    every: int = 1000
    progress_every: int = 25000
    plots: bool = True
    plot_trajectories: int = 12


@dataclass(frozen=True)
class NumericsConfig:
    stop_on_subnormal: bool = True
    support_margin_cap: float = 350.0
    stagnation_patience: int = 1


@dataclass(frozen=True)
class DirectionalConfig:
    model: ModelConfig = ModelConfig()
    initialization: InitializationConfig = InitializationConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    schedule: ScheduleConfig = ScheduleConfig()
    run: RunConfig = RunConfig()
    logging: LoggingConfig = LoggingConfig()
    numerics: NumericsConfig = NumericsConfig()

    def validate(self) -> None:
        def finite(name, value, *, positive=False):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if positive and value <= 0:
                raise ValueError(f"{name} must be positive")

        def integer(name, value, minimum=0):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")

        m, i, o, s, r, log, n = (self.model, self.initialization, self.optimizer,
                                  self.schedule, self.run, self.logging, self.numerics)
        threshold = rho_lower_bound(m)
        finite("model.Lambda", m.Lambda, positive=True)
        finite("initialization.rho", i.rho, positive=True)
        if not isinstance(i.require_admissible_rho, bool):
            raise ValueError("require_admissible_rho must be a boolean")
        if i.require_admissible_rho and i.rho <= threshold:
            raise ValueError(f"initialization.rho must be strictly greater than rho_0={threshold:.17g}")
        z_min, z_max, pi_min, pi_max = i.control_bounds(m)
        for name, value in (("z_min", z_min), ("z_max", z_max), ("pi_min", pi_min), ("pi_max", pi_max)):
            finite("initialization." + name, value, positive=name.startswith("pi"))
        if not 0 <= z_min <= z_max <= 3 * math.log(m.M):
            raise ValueError("z controls must lie within [0, 3*log(M)]")
        if pi_min > pi_max:
            raise ValueError("pi_min must not exceed pi_max")
        try:
            rho_power = i.rho ** m.layers
            a_max = (1 + z_max / rho_power) / m.M
        except (OverflowError, ZeroDivisionError) as error:
            raise ValueError("rho**layers is outside the representable range") from error
        if not math.isfinite(rho_power) or not math.isfinite(a_max) or rho_power <= 0:
            raise ValueError("rho**layers and initialization controls must be representable")
        if not 0 < a_max < 1 / (m.B + 1):
            raise ValueError("initialization controls must give 0 < u1 < u2")
        for name in ("grid_size", "sample_count", "max_trajectories"):
            integer("initialization." + name, getattr(i, name), 1)
        integer("initialization.seed", i.seed)
        if i.sampling not in {"grid", "random", "explicit"}:
            raise ValueError("sampling must be grid, random, or explicit")
        for name, lower, upper in (("z_values", z_min, z_max), ("pi_values", pi_min, pi_max)):
            values = getattr(i, name)
            if values is not None:
                if i.sampling != "grid" or not isinstance(values, (list, tuple)) or not values:
                    raise ValueError(f"{name} requires a nonempty grid array")
                for value in values:
                    finite(name, value)
                    if not lower <= value <= upper:
                        raise ValueError(f"{name} must lie within the configured bounds")
                if any(a >= b for a, b in zip(values, values[1:])):
                    raise ValueError(f"{name} must be strictly ascending")
        if i.sampling == "explicit":
            if not isinstance(i.control_points, (list, tuple)) or not i.control_points:
                raise ValueError("explicit sampling requires control_points")
            seen = set()
            for point in i.control_points:
                if not isinstance(point, (list, tuple)) or len(point) != m.layers:
                    raise ValueError(f"each control point needs {m.layers} coordinates")
                for j, value in enumerate(point):
                    finite("control_points", value)
                    lo, hi = (z_min, z_max) if j == 0 else (pi_min, pi_max)
                    if not lo <= value <= hi:
                        raise ValueError("control_points must lie within the configured bounds")
                if tuple(point) in seen:
                    raise ValueError("control_points must not contain duplicate rows")
                seen.add(tuple(point))
            count = len(i.control_points)
        else:
            if i.control_points is not None:
                raise ValueError("control_points requires explicit sampling")
            count = ((len(i.z_values) if i.z_values is not None else i.grid_size)
                     * (len(i.pi_values) if i.pi_values is not None else i.grid_size) ** (m.layers - 1)
                     if i.sampling == "grid" else i.sample_count)
        if i.sectors not in {"original", "swapped", "both"}:
            raise ValueError("sectors must be original, swapped, or both")
        count *= 2 if i.sectors == "both" else 1
        if count > i.max_trajectories:
            raise ValueError(f"requested {count} trajectories exceeds max_trajectories={i.max_trajectories}")
        for name in ("beta1", "beta2"):
            finite("optimizer." + name, getattr(o, name))
            if not 0 <= getattr(o, name) < 1:
                raise ValueError(f"optimizer.{name} must lie in [0,1)")
        finite("optimizer.epsilon", o.epsilon)
        finite("schedule.initial_rate", s.initial_rate, positive=True)
        finite("schedule.alpha", s.alpha)
        finite("numerics.support_margin_cap", n.support_margin_cap)
        if o.epsilon < 0 or n.support_margin_cap < 0:
            raise ValueError("epsilon and support_margin_cap must be nonnegative")
        integer("run.steps", r.steps)
        integer("run.threads", r.threads, 1)
        if r.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if not isinstance(r.device, str) or not r.device or not isinstance(r.output_dir, str) or not r.output_dir:
            raise ValueError("device and output_dir must be nonempty strings")
        integer("logging.every", log.every, 1)
        integer("logging.progress_every", log.progress_every)
        integer("logging.plot_trajectories", log.plot_trajectories, 1)
        integer("numerics.stagnation_patience", n.stagnation_patience)
        if not isinstance(log.plots, bool) or not isinstance(n.stop_on_subnormal, bool):
            raise ValueError("plots and stop_on_subnormal must be booleans")

    def as_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> DirectionalConfig:
    with Path(path).open("rb") as stream:
        values = tomllib.load(stream)
    sections = {"model": ModelConfig, "initialization": InitializationConfig,
                "optimizer": OptimizerConfig, "schedule": ScheduleConfig,
                "run": RunConfig, "logging": LoggingConfig, "numerics": NumericsConfig}
    if unknown := values.keys() - sections.keys():
        raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
    parsed = {}
    for name, cls in sections.items():
        raw = values.get(name, {})
        if not isinstance(raw, dict):
            raise ValueError(f"{name} must be a TOML table")
        if unknown := raw.keys() - {field.name for field in fields(cls)}:
            raise ValueError(f"unknown {name} keys: {sorted(unknown)}")
        parsed[name] = cls(**raw)
    config = DirectionalConfig(**parsed)
    config.validate()
    return config
