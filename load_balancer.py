"""Compatibility shim for :mod:`llm_loadbalancer.load_balancer`."""

from __future__ import annotations

import sys

from llm_loadbalancer import load_balancer as _impl

if __name__ == "__main__":
    raise SystemExit(_impl.main(sys.argv[1:]))

sys.modules[__name__] = _impl
