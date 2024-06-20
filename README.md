# CS610_Project
CS610 Project 2024

## Running the code
Download the dataset from LUNA16 and place them in this format 
```
-LUNA16{BASE_LOCATION}
--seg-lungs-LUNA16
--annotation.csv {ANNOTATIONS_LOC}
--candidates.csv {CANDIDATES_LOC}
--subset0
--subset1
...
```

1) Run the dataset prep script
```
python3 dataset_prep.py --base_loc {BASE_LOCATION} --save_loc {SAVE_LOCATION}
```
2) Run the training script
```
python3 training_script.py --data_loc {SAVE_LOCATION} --save_loc {MODEL_SAVE LOCATION} --candidates_loc {CANDIDATES_LOC} --batch_size {BATCH_SIZE} --epochs {EPOCHS}
```
3) Display output from display output 
```
python3 display_output.py --model_path {MODEL_PATH} --threshold {THRESHOLD} --dataset_loc {SAVE_LOCATION}

```


