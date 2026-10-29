from common import DATA, fast_thread, path, real_files
import sys, time, cProfile, pstats, xarray as xr, warnings
warnings.simplefilter('ignore')
p = path('wrfout_d01_2025-10-26_06_20_00.nc')
eng = sys.argv[1]
def run():
    t=time.perf_counter(); ds = xr.open_dataset(p, engine=eng); t1=time.perf_counter()
    ds.load(); t2=time.perf_counter(); ds.close()
    return t1-t, t2-t1
run()
print(eng, 'open %.3f load %.3f' % run())
cProfile.run('run()', 'xr.prof')
s = pstats.Stats('xr.prof'); s.sort_stats('cumulative').print_stats(25)
s.sort_stats('tottime').print_stats(12)
