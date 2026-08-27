"""Shared output for the demos — the SDK's teaching surface.

Every demo prints the same two things: a banner naming the structure
and the dimension it ran at, then a right-aligned capacity table of
measurement against prediction. Eighteen modules had each re-derived
that formatting in f-strings, which is why column widths and header
styles had quietly drifted apart.

The API exists to be boring:

    banner("HoloMap: hash map in superposition", dim)
    t = Table(("pairs N", 8), ("load N/d", 9, ".2f"),
              ("pred noise", 11, ".3f"), ("accuracy", 9, ".1%"))
    t.header()
    for ...:
        t.row(n_pairs, n_pairs / dim, noise, correct / n_pairs)

A column is `(heading, width)` or `(heading, width, format)`, where
format is a normal format spec — `.2f`, `.1%`, `,` — applied to the
value and right-aligned in the width. Output is byte-identical to the
hand-rolled f-strings it replaces; that equivalence is pinned by
tests/test_demokit.py against captured demo output.
"""

__all__ = ["Table", "banner"]


def banner(title, dim=None):
    """`== title (d=NNNN) ==`, the first line of every demo."""
    if dim is None:
        print("== %s ==" % title)
    else:
        print("== %s (d=%d) ==" % (title, dim))


class Table:
    """Right-aligned columns with a header, printed row by row.

    Rows print as they are produced rather than being collected: a
    demo that dies partway through should still show what it measured
    before it died.
    """

    def __init__(self, *columns, **kwargs):
        self.indent = kwargs.pop("indent", "")
        if kwargs:
            raise TypeError("unexpected kwargs: %s" % sorted(kwargs))
        self.columns = [(c[0], c[1], c[2] if len(c) > 2 else "")
                        for c in columns]

    def header(self):
        cells = ["{:>{w}}".format(name, w=width)
                 for name, width, _ in self.columns]
        print(self.indent + " ".join(cells))

    def row(self, *values):
        if len(values) != len(self.columns):
            raise ValueError("%d values for %d columns"
                             % (len(values), len(self.columns)))
        cells = ["{:>{w}{f}}".format(value, w=width, f=fmt)
                 for value, (_, width, fmt) in zip(values, self.columns)]
        print(self.indent + " ".join(cells))
