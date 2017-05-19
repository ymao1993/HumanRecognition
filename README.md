# HumanRecognition

The system is develped to perform person recognition task on PIPA dataset, the detailed description of the approach of this system can be found [here](https://www.dropbox.com/s/00ioth4scki918f/ProjectReport_11775.pdf?dl=0).

## Performance (Last Updated: 5/19/2017)

|               Config              | Accuracy |
|:---------------------------------:|:--------:|
|                Face               |  62.18%  |
|                Head               |  63.19%  |
|             Upper-body            |  67.44%  |
|             Full-body             |  58.96%  |
|         All modality fused        |  82.31%  |
| All modality fused + MRF refining |          |

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

### Head Feature 

+ train head feature extractor (feel free to experiment with different batch size)
	
	```
	python pyHumanRecog/head_feature_extractor_train.py --batch_size 32
	```
	
+ test head feature extractor

	```
	python pyHumanRecog/head_feature_extractor_test.py --batch_size 32
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
	
### Upper-body Feature 

+ train upper-body feature extractor (feel free to experiment with different batch size)
	
	```
	python pyHumanRecog/upper_body_feature_extractor_train.py --batch_size 32
	```
	
+ test upper-body feature extractor

	```
	python pyHumanRecog/upper_body_feature_extractor_test.py --batch_size 32
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

To perform MRF optimization (which incorporates the photo-level cooccurrence and mutual exclusive pattern into the final prediction), Set `refine_with_photo_level_context = True` in `HumanRecog/performance_test_config.py`.


