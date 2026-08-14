"""Distributed search: routing, inter-node transport and coordination.

Nothing in this package imports FastAPI. The router, the shard-client protocol
and the coordinator are ordinary Python objects, so the distributed behaviour
can be exercised without a web server — and without containers.
"""
