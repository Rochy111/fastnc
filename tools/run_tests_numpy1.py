"""Run the test suite under numpy 1 (the bench-wrf env: numpy 1.26, numba
0.59), from any Python. Arguments go to pytest.

    python tools/run_tests_numpy1.py                    # the whole suite
    python tools/run_tests_numpy1.py --refs corpus -x   # as CI does, stop at the first failure
    python tools/run_tests_numpy1.py --nep50 ...        # also warn wherever numpy 2 would compute differently

--nep50 sets numpy 1.26's promotion state to 'weak_and_warn' (via
FASTNC_NEP50_WARN, read by tests/conftest.py): every operation whose
result type changes under numpy 2's NEP 50 warns, so such sites in
fastnc are found while numpy 1 is still at hand.

Another interpreter: --python PATH (or $FASTNC_NUMPY1_PYTHON).
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = r'C:\Users\Roch\miniconda3\envs\bench-wrf\python.exe'


def main(argv):
    python = os.environ.get('FASTNC_NUMPY1_PYTHON', DEFAULT)
    env = dict(os.environ)
    args = []
    it = iter(argv)
    for a in it:
        if a == '--python':
            python = next(it)
        elif a == '--nep50':
            env['FASTNC_NEP50_WARN'] = '1'
        else:
            args.append(a)
    if not os.path.exists(python):
        sys.exit('no numpy 1 interpreter at %s (use --python PATH)' % python)
    env.pop('PYTHONPATH', None)
    subprocess.run([python, '-c', 'import numpy, numba, sys; print(sys.version.split()[0], "numpy", '
                    'numpy.__version__, "numba", numba.__version__)'], env=env)
    return subprocess.run([python, '-m', 'pytest'] + (args or ['-q', '-ra']), cwd=ROOT, env=env).returncode


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
