#### Semantic Segmentation
- Download `indoor3d_sem_seg_hdf5_data` from PointNet using the script in the `download_data.sh` script to the current directory.
    ```
    bash download_data.sh
    ```
  This will download indoor3d_sem_seg_hdf5_data (around 1.6GB) to this directory.
  
- For training, we followed the same setup with PointNet. The training is a 6-fold process. Each area will be used 
as a test set, and 6 models will be trained.

    ```
    python train.py --log_dir log/model1 --test_area 1
    python train.py --log_dir log/model2 --test_area 2
    python train.py --log_dir log/model3 --test_area 3
    python train.py --log_dir log/model4 --test_area 4
    python train.py --log_dir log/model5 --test_area 5
    python train.py --log_dir log/model6 --test_area 6
    ```

- For testing, first download the unzipped [S3DIS](https://goo.gl/forms/4SoGp4KtH1jfRqEj2) Dataset, `Stanford3dDataset_v1.2_Aligned_Version`(17GB), into the `data` sub-folder.

- Then, run `python collect_indoor3d_data.py` in the current directory to preprocess the data.

- Run batch_inference.py to segment rooms in each test set. Some OBJ files will be created for prediction visualization in `all_pred_folder`.
    ```
    python batch_inference.py --model_path log/model1/last_model.ckpt --dump_dir all_pred_folder --output_filelist log/model1/output_filelist.txt --room_data_filelist meta/area1_data_label.txt --visu
    python batch_inference.py --model_path log/model2/last_model.ckpt --dump_dir all_pred_folder --output_filelist log/model2/output_filelist.txt --room_data_filelist meta/area2_data_label.txt --visu
    python batch_inference.py --model_path log/model3/last_model.ckpt --dump_dir all_pred_folder --output_filelist log/model3/output_filelist.txt --room_data_filelist meta/area3_data_label.txt --visu
    python batch_inference.py --model_path log/model4/last_model.ckpt --dump_dir all_pred_folder --output_filelist log/model4/output_filelist.txt --room_data_filelist meta/area4_data_label.txt --visu
    python batch_inference.py --model_path log/model5/last_model.ckpt --dump_dir all_pred_folder --output_filelist log/model5/output_filelist.txt --room_data_filelist meta/area5_data_label.txt --visu
    python batch_inference.py --model_path log/model6/last_model.ckpt --dump_dir all_pred_folder --output_filelist log/model6/output_filelist.txt --room_data_filelist meta/area6_data_label.txt --visu
    ```

- To evaluate overall segmentation accuracy, we evaluate all 6 models on their corresponding test areas in `all_pred_folder` and use `eval_iou_accuracy.py` to produce point classification accuracy and IoU.
    ```
    python eval_iou_accuracy.py
    ```