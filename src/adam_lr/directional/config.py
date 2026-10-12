"""Readable, strictly validated configuration for the discrete product model."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
import tomllib
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    layers: int = 3
    B: float = 2.0
    C: float = 1.0
    M: float = 128.0


@dataclass(frozen=True)
class InitializationConfig:
    rho: float = 1.0
    sampling: str = "grid"  # grid: n**L controls; random: sample_count controls
    grid_size: int = 11
    sample_count: int = 256
    seed: int = 0
    s_min: float = 0.0
    s_max: float = 3 * math.log(128.0)
    ell_min: float = -1.65
    ell_max: float = 1.65
    sectors: str = "both"  # original, swapped, both (also swaps the data)
    max_trajectories: int = 100_000


@dataclass(frozen=True)
class OptimizerConfig:
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 0.0


@dataclass(frozen=True)
class ScheduleConfig:
    initial_rate: float = 0.01
    power: float = 0.0  # eta_k = initial_rate * k**(-power), k starts at 1


@dataclass(frozen=True)
class RunConfig:
    steps: int = 400_000
    dtype: str = "float64"
    device: str = "cpu"
    threads: int = 1
    output_dir: str = "runs/directional/final_rho1_eta001"


@dataclass(frozen=True)
class LoggingConfig:
    every: int = 5000
    geometric_checkpoints: int = 80
    progress_every: int = 10000
    plots: bool = True
    plot_trajectories: int = 12


@dataclass(frozen=True)
class NumericsConfig:
    # With positive data and parameters every exact gradient coordinate is negative.
    # These guards stop at the first unusable proposal, preserving the preceding state.
    stop_on_subnormal: bool = True
    support_margin_cap: float = 250.0  # 0 disables this user-selected horizon
    stagnation_patience: int = 0  # 0 disables; a stop is never called convergence


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
        def finite(name: str, value: float, *, positive: bool = False) -> None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if positive and value <= 0:
                raise ValueError(f"{name} must be positive")

        def integer(name: str, value: int, minimum: int = 0) -> None:
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")

        m, i, o, s, r, log, n = (self.model, self.initialization, self.optimizer,
                                  self.schedule, self.run, self.logging, self.numerics)
        integer("model.layers", m.layers, 2)
        for name in ("B", "C", "M"):
            finite(f"model.{name}", getattr(m, name), positive=True)
        finite("initialization.rho", i.rho, positive=True)
        for name in ("s_min", "s_max", "ell_min", "ell_max"):
            finite(f"initialization.{name}", getattr(i, name))
        if not 0 <= i.s_min <= i.s_max:
            raise ValueError("initialization controls require 0 <= s_min <= s_max")
        if i.ell_min > i.ell_max:
            raise ValueError("ell_min must not exceed ell_max")
        integer("initialization.grid_size", i.grid_size, 1)
        integer("initialization.sample_count", i.sample_count, 1)
        integer("initialization.seed", i.seed)
        integer("initialization.max_trajectories", i.max_trajectories, 1)
        if i.sampling not in {"grid", "random"}:
            raise ValueError("sampling must be 'grid' or 'random'")
        if i.sectors not in {"original", "swapped", "both"}:
            raise ValueError("sectors must be original, swapped, or both")
        count = i.grid_size ** m.layers if i.sampling == "grid" else i.sample_count
        count *= 2 if i.sectors == "both" else 1
        if count > i.max_trajectories:
            raise ValueError(f"requested {count} trajectories exceeds max_trajectories={i.max_trajectories}")
        try:
            rho_power = i.rho ** m.layers
            a_max = (1 + i.s_max / rho_power) / m.M
        except (OverflowError, ZeroDivisionError) as error:
            raise ValueError("rho**layers is outside the representable range") from error
        if not math.isfinite(rho_power) or not math.isfinite(a_max) or rho_power <= 0:
            raise ValueError("rho**layers and initial controls must be finite and positive")
        if not 0 < a_max < 1 / (m.B + 1):
            raise ValueError("initial controls must satisfy (1+s_max/rho**L)/M < 1/(B+1), giving 0<u1<u2")
        for name in ("beta1", "beta2"):
            finite(f"optimizer.{name}", getattr(o, name))
            if not 0 <= getattr(o, name) < 1:
                raise ValueError(f"optimizer.{name} must lie in [0,1)")
        finite("optimizer.epsilon", o.epsilon)
        if o.epsilon < 0:
            raise ValueError("epsilon must be nonnegative")
        finite("schedule.initial_rate", s.initial_rate, positive=True)
        finite("schedule.power", s.power)
        integer("run.steps", r.steps)
        integer("run.threads", r.threads, 1)
        if r.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if not isinstance(r.device, str) or not r.device:
            raise ValueError("device must be a nonempty torch device string")
        if not isinstance(r.output_dir, str) or not r.output_dir:
            raise ValueError("output_dir must be nonempty")
        integer("logging.every", log.every)
        integer("logging.geometric_checkpoints", log.geometric_checkpoints)
        integer("logging.progress_every", log.progress_every)
        integer("logging.plot_trajectories", log.plot_trajectories, 1)
        if not isinstance(log.plots, bool) or not isinstance(n.stop_on_subnormal, bool):
            raise ValueError("plots and stop_on_subnormal must be booleans")
        integer("numerics.stagnation_patience", n.stagnation_patience)
        finite("numerics.support_margin_cap", n.support_margin_cap)
        if n.support_margin_cap < 0:
            raise ValueError("support_margin_cap must be nonnegative")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> DirectionalConfig:
    """Load TOML with unknown sections/keys rejected (avoids misspelled overrides)."""
    with Path(path).open("rb") as handle:
        values = tomllib.load(handle)
    sections = {"model": ModelConfig, "initialization": InitializationConfig,
                "optimizer": OptimizerConfig, "schedule": ScheduleConfig,
                "run": RunConfig, "logging": LoggingConfig, "numerics": NumericsConfig}
    if unknown := values.keys() - sections.keys():
        raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
    parsed = {}
    for name, cls in sections.items():
        section = values.get(name, {})
        if not isinstance(section, dict):
            raise ValueError(f"{name} must be a TOML table")
        if unknown := section.keys() - {f.name for f in fields(cls)}:
            raise ValueError(f"unknown {name} keys: {sorted(unknown)}")
        parsed[name] = cls(**section)
    config = DirectionalConfig(**parsed)
    config.validate()
    return config
