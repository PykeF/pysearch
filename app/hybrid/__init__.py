"""Hybrid retrieval: combining the lexical and semantic rankings.

Nothing here retrieves anything. The two retrieval systems already exist and are
correct on their own; this package only decides how to merge their output, which
is why it is a pure function over two ranked lists and imports neither FastAPI
nor any retrieval machinery.
"""
