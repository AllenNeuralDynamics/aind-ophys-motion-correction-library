"""Suite2P motion correction for AIND optical physiology.

Heavy entry points are imported lazily from their submodules
(``.job.run``) so the light metadata/QC/settings modules stay importable
without the suite2p/torch stack.
"""
