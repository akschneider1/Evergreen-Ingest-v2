import dataclasses
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import langextract as lx


@dataclasses.dataclass
class DomainConfig:
    name: str
    prompt: str
    examples: list  # list[lx.data.ExampleData]


# Registry populated by each domain module
DOMAINS: dict[str, DomainConfig] = {}

# Load domains lazily to avoid import errors when langextract is unavailable
def _load_all():
    for module_name in ("examples.tax", "examples.benefits"):
        importlib.import_module(module_name)


def get_domain(name: str) -> DomainConfig:
    if not DOMAINS:
        _load_all()
    if name not in DOMAINS:
        raise ValueError(f"Unknown domain: {name!r}. Available: {list(DOMAINS)}")
    return DOMAINS[name]
