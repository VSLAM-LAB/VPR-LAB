# VPR-LAB

Create distance matrix:
```bash
python3 main.py --method=megaloc --backbone=ResNet18 --descriptors_dimension=512 \
    --database_folder=/home/alejandro/VSLAM-LAB-Benchmark/ANTARCTICA/DJI_202401111904_001/rgb --queries_folder=/home/alejandro/VSLAM-LAB-Benchmark/ANTARCTICA/DJI_202401111904_001/rgb \
    --no_labels --log_dir /home/alejandro/ICRA26/Experiments/test
```

Create ground truth matrix:
```bash
python create_groundtruth_matrix.py
```

Create mask matrix:
```bash
python create_mask_matrix.py
```

