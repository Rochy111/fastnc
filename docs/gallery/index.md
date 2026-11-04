---
hide:
  - toc
---

# Gallery

Each example is a complete script with the figure it produces. The data are read with fastnc's three interfaces (the netCDF4-style `Dataset`, the h5py-style `File` and the xarray engine) and drawn with [fastplot](https://rochy111.github.io/fastplot/). The scripts are in `docs/examples/`.

## Satellites

<div class="cards">
<a class="card" href="goes_full_disk/">
<div class="thumb"><img src="../images/goes_full_disk.png" alt="GOES-19 full disk" loading="lazy"></div>
<div class="body"><span class="title">GOES-19 full disk</span><span class="note">Band 13 brightness temperature, with Dataset</span></div>
</a>
<a class="card" href="viirs_window/">
<div class="thumb"><img src="../images/viirs_window.png" alt="A hurricane's eye at 375 m" loading="lazy"></div>
<div class="body"><span class="title">A hurricane's eye at 375 m</span><span class="note">A window of VIIRS I5, with File</span></div>
</a>
<a class="card" href="modis_melissa/">
<div class="thumb"><img src="../images/modis_melissa.png" alt="Hurricane Melissa from MODIS" loading="lazy"></div>
<div class="body"><span class="title">Hurricane Melissa from MODIS</span><span class="note">An HDF-EOS swath on a map, with File</span></div>
</a>
<a class="card" href="avhrr_andrew/">
<div class="thumb"><img src="../images/avhrr_andrew.png" alt="Hurricane Andrew, 1992" loading="lazy"></div>
<div class="body"><span class="title">Hurricane Andrew, 1992</span><span class="note">A NOAA-12 AVHRR pass, with Dataset</span></div>
</a>
</div>

## Models and speed

<div class="cards">
<a class="card" href="wrf_melissa/">
<div class="thumb"><img src="../images/wrf_melissa.png" alt="Hurricane Melissa in WRF" loading="lazy"></div>
<div class="body"><span class="title">Hurricane Melissa in WRF</span><span class="note">10 m wind, with the xarray engine</span></div>
</a>
<a class="card" href="speed_chart/">
<div class="thumb"><img src="../images/speed_chart.png" alt="Benchmark chart" loading="lazy"></div>
<div class="body"><span class="title">Benchmark chart</span><span class="note">Reading every variable against netCDF4-python and h5py</span></div>
</a>
</div>
