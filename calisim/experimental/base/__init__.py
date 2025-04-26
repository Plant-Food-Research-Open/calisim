import importlib

__all__ = []

if importlib.util.find_spec("elfi") is not None:
	from .elfi_base import ELFIBase

	__all__.append(ELFIBase)
