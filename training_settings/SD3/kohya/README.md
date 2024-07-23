Running on this `sd-scripts` commit hash: `082f13658bdbaed872ede6c0a7a75ab1a5f3712d`

To run with restarts, look at this sample command:

```
python /home/pure_water_100/sd-scripts/sd3_train.py --config_file /home/pure_water_100/kohya_train/sd3_medium_full_11/config/adam_full_finetune_sd3_11.toml --lr_scheduler_type CosineAnnealingLR --lr_scheduler_args T_max=2382
```