"""Limites operacionais nao devem ser interpretados como falhas causais."""
from dataclasses import dataclass, field
from time import perf_counter


class PredictiveOperationalTimeout(TimeoutError):
    pass


@dataclass
class AnalysisTimer:
    max_total_seconds: float = 240
    max_static_stage_seconds: float = 60
    seconds: dict[str, float] = field(default_factory=dict)
    started: float = field(default_factory=perf_counter)

    def measure(self, name, function, *args, **kwargs):
        start = perf_counter()
        result = function(*args, **kwargs)
        elapsed = perf_counter() - start
        self.seconds[name] = self.seconds.get(name, 0) + elapsed
        # Cooperative stage boundaries. The evaluation worker enforces hard
        # wall/memory limits even if AST/model processing has not returned.
        if (perf_counter() - self.started > self.max_total_seconds
                or (name != 'qwen_analysis' and elapsed > self.max_static_stage_seconds)):
            raise PredictiveOperationalTimeout(f"NOT_EVALUATED_TIMEOUT: {name}")
        return result
