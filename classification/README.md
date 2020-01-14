#### Shape Classification
- Download `modelnet40_ply_hdf5_2048` from PointNet using the script in the `data` folder.
    ```
    cd data
    bash download_data.sh
    cd ..
    ```
  This will download point clouds with 2048 points uniformly sampled from a shape surface and normalized with zero-mean by [pointnet](https://github.com/charlesq34/pointnet) (417MB).
  By default, 1024 points will be used for training and testing from this dataset.
  
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
  To test a pretrained model, run `download_checkpoint.sh` in checkpoints, then run:
    ```
    python test.py --model_path checkpoints/model.ckpt
    ```
