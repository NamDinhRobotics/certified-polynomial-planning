"""Resolve an immutable executed source by its recorded SHA256, without weakening checks."""
from pathlib import Path
import hashlib

def checked_source(root,name,expected):
    root=Path(root)
    for p in (root/name,root/'provenance'/Path(name).name,
              root/'executed_sources'/expected/Path(name).name):
        if p.is_file() and p.resolve().is_relative_to(root.resolve()) and hashlib.sha256(p.read_bytes()).hexdigest()==expected:
            return p
    raise AssertionError('Missing exact executed source: '+name+' '+expected)
