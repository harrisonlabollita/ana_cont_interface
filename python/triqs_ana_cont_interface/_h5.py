# Copyright (c) 2026 Harrison LaBollita
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You may obtain a copy of the License at
#     https://www.gnu.org/licenses/gpl-3.0.txt
#
# Authors: Harrison LaBollita

"""HDF5 registration for the package's dataclass records."""

import dataclasses

try:
    from h5.formats import register_class
except ImportError:  # h5 is only needed for archiving
    register_class = None


def _encode(value):
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {str(k): _encode(v) for k, v in value.items()}
    return value


def register_dataclass(cls):
    """Make a dataclass HDF5-serializable through the TRIQS h5 protocol.

    Fields that are None are omitted on write and restored as None on read,
    so adding an optional field does not invalidate existing archives.
    """
    # namespaced: the h5 format registry is process-global and shared
    # with every other TRIQS package in the session
    cls._hdf5_format_ = "AnaContInterface" + cls.__name__

    def reduce_to_dict(self):
        out = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if v is None:
                continue
            out[f.name] = _encode(v)
        return out

    def factory_from_dict(kls, name, d):
        kwargs = {f.name: d.get(f.name) for f in dataclasses.fields(kls)}
        return kls(**kwargs)

    cls.__reduce_to_dict__ = reduce_to_dict
    cls.__factory_from_dict__ = classmethod(factory_from_dict)
    if register_class is not None:
        register_class(cls)
    return cls
