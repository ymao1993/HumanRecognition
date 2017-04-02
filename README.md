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
	