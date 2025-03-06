# Real-Time Detector Alignment

## Generate Reduced-Order Model Detector Data
- folder: `./rtal/data`
- How to run:
  ```
  python ./rtal/data/generate.py --num-samples 1000 --config [path/to/config_file] --output-folder [path/to/dataset_folder]
  ```
<p align="center">
  <img src="https://github.com/abidihaider/RealTimeAlignment/blob/main/notebooks/plots/plot_3d_1_start.png" width="45%" title="Reduced-Order Model data, 3D">
  <img src="https://github.com/abidihaider/RealTimeAlignment/blob/main/notebooks/plots/plot_3d_1_curr.png" width="45%" title="Reduced-Order Model data, 3D">
</p>
<p align="center">
  <img src="https://github.com/abidihaider/RealTimeAlignment/blob/main/notebooks/plots/plot_2d_1_from-readout.png" width="95%" title="Reduced-Order Model data, 2D">
</p>
## Dataset API for loading ROM data (TBD)
## Model (TBD)


## Archived
```
  python3 generateData.py --nEvents 10 --outputName output.json
```
