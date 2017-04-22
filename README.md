# HumanRecognition

## Usage
+ get dataset

	```
	./scripts/get_PIPA.sh
	```

+ get models

	```
	./scripts/get_models.sh
	```
+ extract CPM pose

	```
	python pyHumanRecog/extract_pose.py <img_dump_folder> <pose_dump_folder>
	```
	*\<image\_dump\_folder\>*: folder to dump CPM pose visualization images

	*\<pose\_dump\_folder\>*: folder to dump CPM pose positions
	
+ train body feature extractor

	```
	mkdir body_model
	mkdir body_log
	python pyHumanRecog/train_body_feature_extractor.py --model_save_dir body_model --summary_dir body_log
	
	```


	