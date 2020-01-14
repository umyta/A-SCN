#### Part Segmentation
- Download `hdf5_data` and `PartAnnotation` from PointNet using the script in the `data` folder.
    ```
    cd data
    bash download_data.sh
    cd ..
    ```
  This will download and create `ShapeNetPart` (around 3.8GB) and `hdf5_data` (around 347MB) folders.
  
- For training, run the following code:
    
    ```
    python train.py
    ```
  To see HELP for the training script:
  
  ```
  python train.py -h
  ```
  
- For testing, run:

    ```
    python test.py
    ```