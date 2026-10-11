"""Limites operacionais nao devem ser interpretados como falhas causais."""
from dataclasses import dataclass, field
from time import perf_counter
from contextlib import contextmanager
from contextvars import ContextVar


_checkpoint_observer = ContextVar('predictive_checkpoint_observer', default=None)


@contextmanager
def observe_checkpoints(observer):
    """O consumidor de observabilidade persiste; o predictive nao recebe writer."""
    token = _checkpoint_observer.set(observer)
    try:
        yield
    finally:
        _checkpoint_observer.reset(token)


def observe_phase(name, function, *args, **kwargs):
    observer = _checkpoint_observer.get()
    start = perf_counter()
    if observer:
        observer({'phase': name, 'state': 'START', 'elapsed_seconds': 0.0})
    try:
        result = function(*args, **kwargs)
    except BaseException as error:
        if observer:
            observer({'phase': name, 'state': 'ERROR', 'elapsed_seconds': perf_counter()-start,
                      'error_type': type(error).__name__})
        raise
    if observer:
        observer({'phase': name, 'state': 'END', 'elapsed_seconds': perf_counter()-start})
    return result


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
        result = observe_phase(name, function, *args, **kwargs)
        elapsed = perf_counter() - start
        self.seconds[name] = self.seconds.get(name, 0) + elapsed
        # Cooperative stage boundaries. The evaluation worker enforces hard
        # wall/memory limits even if AST/model processing has not returned.
        if (perf_counter() - self.started > self.max_total_seconds
                or (name != 'qwen_analysis' and elapsed > self.max_static_stage_seconds)):
            raise PredictiveOperationalTimeout(f"NOT_EVALUATED_TIMEOUT: {name}")
        return result
