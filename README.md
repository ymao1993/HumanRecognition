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

### Face Feature

We use [FaceNet](https://arxiv.org/abs/1503.03832) for face feature extraction. FaceNet is a CNN trained to directly optimize the embedding itself.

+ test face feature extractor

	```
	python pyHumanRecog/face_feature_extractor_test.py
	```

### Body Feature 

+ train body feature extractor (feel free to experiment with different batch size)
	
	```
	python pyHumanRecog/body_feature_extractor_train.py --batch_size 32
	```
	
+ test body feature extractor

	```
	python pyHumanRecog/body_feature_extractor_test.py --batch_size 32
	```
	
### Pose estimation

We use CPM for pose estimation. The estimated CPM pose will mainly be used for image warping.

+ CPM pose estimation
	
	```
	python pyHumanRecog/extract_pose.py <img_dump_folder> <pose_dump_folder>
	```
	`<image_dump_folder>`: folder to dump CPM pose visualization images

	`<pose_dump_folder>`: folder to dump CPM pose positions

### Evaluation

For performance evaluation, Please first modify `performance_test_config.py` (within `pyHumanRecog` folder) to specify the features you wish to use and their corresponding weights. Then execute the following command.

```
python pyHumanRecog/performance_test.py
```

	
## Current Performance

**Last Updated: 05/05/2017**

```
fused model accuracy: 62.11%
accuracy with only body features: 1.48%
accuracy with only face features: 61.90%
accuracy by chance: 0.17%
```
