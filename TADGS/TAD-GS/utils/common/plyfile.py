import numpy as np

_PLY_TO_NUMPY = {
    "float": "<f4", "float32": "<f4", "f4": "<f4",
    "double": "<f8", "float64": "<f8", "f8": "<f8",
    "uchar": "|u1", "uint8": "|u1", "u1": "|u1",
    "char": "|i1", "int8": "|i1", "i1": "|i1",
    "ushort": "<u2", "uint16": "<u2", "u2": "<u2",
    "short": "<i2", "int16": "<i2", "i2": "<i2",
    "uint": "<u4", "uint32": "<u4", "u4": "<u4",
    "int": "<i4", "int32": "<i4", "i4": "<i4",
}

_NUMPY_TO_PLY = {
    "<f4": "float", "=f4": "float", "f4": "float",
    "<f8": "double", "=f8": "double", "f8": "double",
    "|u1": "uchar", "u1": "uchar",
    "|i1": "char", "i1": "char",
    "<u2": "ushort", "=u2": "ushort", "u2": "ushort",
    "<i2": "short", "=i2": "short", "i2": "short",
    "<u4": "uint", "=u4": "uint", "u4": "uint",
    "<i4": "int", "=i4": "int", "i4": "int",
}


class _PlyProperty:
    def __init__(self, name, ply_type):
        self.name = name
        self.ply_type = ply_type


class PlyElement:
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.count = len(data)
        self.properties = [
            _PlyProperty(field_name, _NUMPY_TO_PLY.get(data.dtype.fields[field_name][0].str, "float"))
            for field_name in data.dtype.names
        ]

    @classmethod
    def describe(cls, data, name):
        return cls(name, data)

    def __getitem__(self, key):
        return self.data[key]


class PlyData:
    def __init__(self, elements, text=False):
        self.elements = list(elements)
        self.text = text
        self._by_name = {el.name: el for el in self.elements}

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.elements[key]
        return self._by_name[key]

    def write(self, path):
        with open(path, "wb") as f:
            f.write(b"ply\n")
            f.write(b"format binary_little_endian 1.0\n")
            for el in self.elements:
                f.write(f"element {el.name} {el.count}\n".encode("ascii"))
                for prop in el.properties:
                    f.write(f"property {prop.ply_type} {prop.name}\n".encode("ascii"))
            f.write(b"end_header\n")
            for el in self.elements:
                arr = np.ascontiguousarray(el.data.astype(el.data.dtype.newbyteorder("<"), copy=False))
                f.write(arr.tobytes())

    @classmethod
    def read(cls, path):
        with open(path, "rb") as f:
            line = f.readline().decode("ascii").strip()
            if line != "ply":
                raise ValueError(f"Not a PLY file: {path}")
            fmt = None
            elements_spec = []
            current_el = None
            while True:
                raw = f.readline()
                if not raw:
                    raise ValueError("Unexpected EOF in PLY header")
                s = raw.decode("ascii").strip()
                if s == "end_header":
                    break
                parts = s.split()
                if not parts or parts[0] == "comment":
                    continue
                if parts[0] == "format":
                    fmt = parts[1]
                elif parts[0] == "element":
                    current_el = {"name": parts[1], "count": int(parts[2]), "props": []}
                    elements_spec.append(current_el)
                elif parts[0] == "property" and current_el is not None:
                    current_el["props"].append((parts[2], parts[1]))

            elements = []
            for spec in elements_spec:
                dtype = [(name, _PLY_TO_NUMPY[ptype]) for name, ptype in spec["props"]]
                if fmt == "ascii":
                    rows = []
                    for _ in range(spec["count"]):
                        vals = f.readline().decode("ascii").strip().split()
                        rows.append(tuple(vals))
                    data = np.array(rows, dtype=dtype)
                else:
                    data = np.fromfile(f, dtype=np.dtype(dtype), count=spec["count"])
                elements.append(PlyElement(spec["name"], data))
        return cls(elements, text=(fmt == "ascii"))

