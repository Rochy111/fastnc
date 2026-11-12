"""Dataset.tocdl(): the CDL header ncdump -h prints, made by fastnc itself.
Against ncdump's output saved beside the test files (tests/data/cdl), and
against ncdump itself where it is installed."""
import glob
import os
import shutil
import subprocess

import pytest

import fastnc

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
SAVED = sorted(glob.glob(os.path.join(DATA, 'cdl', '*.cdl')))


def _source(cdl):
    name = os.path.basename(cdl)[:-4]
    for sub in ('', 'hdf4', 'complex', 'filters_hdf'):
        p = os.path.join(DATA, sub, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def test_anonymous_types_named_as_netcdf_c():
    # (h5py's types have no committed datatype: netCDF-c names them, one count over the
    # file in the order it reads it, equal types once, byte order aside)
    with fastnc.Dataset(os.path.join(DATA, 'complex', 'anonymous.h5')) as d:
        assert list(d.cmptypes) == ['_AnonymousCompound2', '_AnonymousCompound3', '_AnonymousCompound4']
        assert list(d.enumtypes) == ['_AnonymousEnum1'] and list(d.vltypes) == ['_AnonymousVlen5']
        assert d['d_be'].datatype.name == d['e_c16'].datatype.name == '_AnonymousCompound4'
        assert d['g/again'].datatype.name == '_AnonymousCompound2'
        assert list(d['g'].cmptypes) == ['_AnonymousCompound6']


def test_text_attribute_with_nul():
    with fastnc.Dataset(os.path.join(DATA, 'hdf4', 'types.hdf')) as d:
        h = d.getncattr('history')
        assert type(h) is str and '\x00' not in h          # (as netCDF4-python gives it)
        assert ':history = "made by hdf4_make_corpus.py\\000with a NUL" ;' in d.tocdl()


@pytest.mark.parametrize('cdl', SAVED, ids=os.path.basename)
def test_as_ncdump_saved(cdl):
    with open(cdl, encoding='utf-8') as f:
        want = f.read().replace('\r\n', '\n')
    with fastnc.Dataset(_source(cdl)) as d:
        assert d.tocdl() == want


def test_outfile_and_refusals(tmp_path):
    p = os.path.join(DATA, 'nc3_classic.nc')
    out = str(tmp_path / 'x.cdl')
    with fastnc.Dataset(p) as d:
        assert d.tocdl(outfile=out) is None
        with open(out, encoding='utf-8') as f:
            assert f.read() == d.tocdl()
        with pytest.raises(NotImplementedError):
            d.tocdl(data=True)


NCDUMP = shutil.which('ncdump')


@pytest.mark.skipif(NCDUMP is None, reason='no ncdump')
@pytest.mark.parametrize('name', ['nc3_classic.nc', 'nc3_cdf5.nc', 'nc3_64bit.nc', 'nc4.nc', 'nc4_classic.nc',
                                  'groups_new.h5', 'groups_old.h5', 'filters.h5', 'idx_latest.h5'])
def test_as_ncdump_live(name):
    p = os.path.join(DATA, name)
    r = subprocess.run([NCDUMP, '-h', p], capture_output=True)
    if r.returncode:
        pytest.skip('ncdump cannot read it')
    with fastnc.Dataset(p) as d:
        assert d.tocdl() == r.stdout.decode('utf-8').replace('\r\n', '\n')
