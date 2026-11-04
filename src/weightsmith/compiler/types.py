"""Static types of Weightsmith expressions.

Types fall into four kinds (see docs/semantics.md):

* structural: ``WeightSet``, ``Model``, ``Params``, ``Selection``. Derivable from files.
* epistemic: ``Behavior`` here; ``Finding`` and ``Evidence`` are runtime values that v0.1 syntax
  cannot name directly.
* transformational: ``Patch`` is produced by the runtime and never named in v0.1 syntax.
* specification and scalar types: ``Int``, ``Float``, ``Fraction``, ``String``, ``Bool``,
  ``List[T]`` and ``Symbol[registry]`` for names drawn from a registry (finders, metrics, ...).

Structural types carry the *binding* they are rooted in, so the checker can tell a selection of the
target apart from a selection of a donor model.
"""

from __future__ import annotations

from dataclasses import dataclass


class Type:
    """Base class of all static types."""

    def __str__(self) -> str:  # pragma: no cover - overridden
        return type(self).__name__


@dataclass(frozen=True, slots=True)
class Prim(Type):
    name: str  # Int | Float | Fraction | String | Bool

    def __str__(self) -> str:
        return self.name


INT = Prim("Int")
FLOAT = Prim("Float")
FRACTION = Prim("Fraction")
STRING = Prim("String")
BOOL = Prim("Bool")


@dataclass(frozen=True, slots=True)
class WeightsType(Type):
    """A loaded weight set. ``is_model`` means an architecture is attached, so it can run."""

    binding: str | None
    is_model: bool

    def __str__(self) -> str:
        kind = "Model" if self.is_model else "WeightSet"
        return f"{kind}({self.binding})" if self.binding else kind


@dataclass(frozen=True, slots=True)
class ParamsType(Type):
    """``M.params``: the parameter namespace of a weight set, indexed by glob patterns."""

    binding: str

    def __str__(self) -> str:
        return f"Params({self.binding})"


@dataclass(frozen=True, slots=True)
class SelectionType(Type):
    """A set of parameters (optionally restricted to index ranges) of one weight set."""

    binding: str
    sliced: bool

    def __str__(self) -> str:
        return f"Selection({self.binding}{', sliced' if self.sliced else ''})"


@dataclass(frozen=True, slots=True)
class BehaviorType(Type):
    """A declared behaviour: a probe suite plus a metric. Not a claim about mechanism."""

    def __str__(self) -> str:
        return "Behavior"


@dataclass(frozen=True, slots=True)
class ListType(Type):
    elem: Type

    def __str__(self) -> str:
        return f"List[{self.elem}]"


@dataclass(frozen=True, slots=True)
class SymbolType(Type):
    """A bare name that refers to an entry of a registry, such as a finder or a metric."""

    registry: str

    def __str__(self) -> str:
        return f"Symbol[{self.registry}]"


ANY_WEIGHTS = WeightsType(None, is_model=False)
ANY_MODEL = WeightsType(None, is_model=True)


def is_assignable(src: Type, dst: Type) -> bool:
    """Whether a value of type ``src`` may be used where ``dst`` is expected."""
    if src == dst:
        return True
    if isinstance(dst, Prim) and isinstance(src, Prim):
        if dst == FLOAT:
            return src in (INT, FRACTION)
        return False
    if isinstance(dst, WeightsType) and isinstance(src, WeightsType):
        binding_ok = dst.binding is None or dst.binding == src.binding
        return binding_ok and (src.is_model or not dst.is_model)
    if isinstance(dst, ListType) and isinstance(src, ListType):
        return is_assignable(src.elem, dst.elem)
    return False
