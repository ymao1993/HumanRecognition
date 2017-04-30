# HumanRecognition

## Usage

### Preparation

+ get dataset

	```
	./scripts/get_PIPA.sh
	```

+ get models

	```
	./scripts/get_models.sh
	```

+ get features (if you don't wish to extract them)

	```
	./scripts/get_features.sh
	```
	
+ extract CPM pose

	```
	python pyHumanRecog/extract_pose.py <img_dump_folder> <pose_dump_folder>
	```
	`<image_dump_folder>`: folder to dump CPM pose visualization images

	`<pose_dump_folder>`: folder to dump CPM pose positions

### Body Feature 

+ train body feature extractor

	(feel free to increase the batch\_size for better performance)
	
	```
	python pyHumanRecog/body_feature_extractor_train.py --batch_size 32
	```
	
+ test body feature extractor

	```
	python pyHumanRecog/body_feature_extractor_test.py --batch_size 32
	```

### Evaluation

For performance evaluation, Please first modify `performance_test_config.py` (within `pyHumanRecog` folder) to specify the features you wish to use and their corresponding weights. Then execute the following command.

```
python pyHumanRecog/performance_test.py
```

	
## Current Performance

**Last Updated: 04/29/2017**

```
total accuracy: 1.48%
accuracy with body: 1.48%
accuracy by chance: 0.17%
```