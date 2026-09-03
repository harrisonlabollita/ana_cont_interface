"""Result records."""

from dataclasses import dataclass

from ._h5 import register_dataclass


@register_dataclass
@dataclass
class TaskDiagnostics:
    """Per-task numbers needed to judge the continuation."""

    block: str
    i: int
    j: int
    alpha_opt: float = None
    chi2: float = None
    chi2_min: float = None
    entropy: float = None
    norm: float = None
    blur_width: float = None
    n_sv: int = None
    shift: complex = 0.0
    alpha_scan: object = None
    im_axis: object = None
    im_data: object = None
    error: object = None
    backtransform: object = None

    @property
    def key(self):
        return (self.block, self.i, self.j)


def _lookup(records, block, i, j):
    for record in records:
        if record.key == (block, i, j):
            return record
    raise KeyError((block, i, j))


@register_dataclass
@dataclass
class ContinuationResult:
    g_w: object
    a_w: object
    diagnostics: tuple
    problem: object = None

    def __post_init__(self):
        self.diagnostics = tuple(self.diagnostics)

    def diag(self, block, i=0, j=0):
        return _lookup(self.diagnostics, block, i, j)


@register_dataclass
@dataclass
class SigmaResult:
    sigma_w: object
    sigma_inf: dict
    first_moment: dict
    aux: ContinuationResult

    @property
    def diagnostics(self):
        return self.aux.diagnostics

    def diag(self, block, i=0, j=0):
        return self.aux.diag(block, i, j)
