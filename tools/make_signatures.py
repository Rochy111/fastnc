"""Write fastnc/_signatures.py: the argument types each kernel Python calls is
used with, taken from this machine's image, so that a fresh install's build
can compile them all before anything needs them.

    python tools/make_signatures.py

Run it after a full test run (which compiles what reads use) and a build,
when kernels or their arguments changed. A signature it no longer finds
matching is skipped by the build, and compiled when first needed.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from fastnc import _image  # noqa: E402


def main():
    im = _image.Image(_image.current())
    table = {}
    out = {}
    for name in sorted(im.kernels):
        out[name] = [[table.setdefault(d, len(table)) for d in key] for key in im.kernels[name]]
    with open(os.path.join(ROOT, 'fastnc', '_signatures.py'), 'w', newline='\n') as f:
        f.write('"""Made by tools/make_signatures.py: the argument types of the kernels Python\n'
                'calls, as _image describes them. Numbers index TYPES."""\n')
        f.write('TYPES = [\n')
        for d in table:
            f.write('    %r,\n' % (d,))
        f.write(']\n\nSIGNATURES = {\n')
        for name, sigs in out.items():
            f.write('    %r: [\n' % name)
            for s in sigs:
                f.write('        %s,\n' % repr(tuple(s)).replace(' ', ''))
            f.write('    ],\n')
        f.write('}\n')
    print(len(table), 'types,', sum(len(s) for s in out.values()), 'signatures of', len(out), 'kernels')


if __name__ == '__main__':
    main()
