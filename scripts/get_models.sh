mkdir -p models

echo 'downloading CPM models...'
mkdir -p models/CPM/_trained_person_MPI/
mkdir -p models/CPM/_trained_MPI/
wget --no-check-certificate https://www.dropbox.com/s/d5x49xnhjiru2af/params.pkl?dl=0 --output-document=models/CPM/_trained_person_MPI/params.pkl
wget --no-check-certificate https://www.dropbox.com/s/uvhkd1l8v65u8ed/params.pkl?dl=0 --output-document=models/CPM/_trained_MPI/params.pkl

echo 'downloading pretrained inception_v3 model for body...'
mkdir -p pretrained_model_body2
wget --no-check-certificate https://www.dropbox.com/s/wtladpcusvcnako/model.ckpt-75030.data-00000-of-00001?dl=0 --output-document=pretrained_model_body2/model.ckpt-75030.data-00000-of-00001
wget --no-check-certificate https://www.dropbox.com/s/fskgr1mf34fst5b/model.ckpt-75030.index?dl=0 --output-document=pretrained_model_body2/model.ckpt-75030.index
wget --no-check-certificate https://www.dropbox.com/s/hefwy1iu9lz82n0/model.ckpt-75030.meta?dl=0 --output-document=pretrained_model_body2/model.ckpt-75030.meta

echo 'downloading pretrained inception_v3 model for upper_body...'
mkdir -p pretrained_model_upperbody
wget --no-check-certificate https://www.dropbox.com/s/m6nmdnxsot22mrk/model.ckpt-104956.data-00000-of-00001?dl=0 --output-document=pretrained_model_upperbody/model.ckpt-104956.data-00000-of-00001
wget --no-check-certificate https://www.dropbox.com/s/4y4dcppm3rn35al/model.ckpt-104956.index?dl=0 --output-document=pretrained_model_upperbody/model.ckpt-104956.index
wget --no-check-certificate https://www.dropbox.com/s/f19nz1j5owxbuc6/model.ckpt-104956.meta?dl=0 --output-document=pretrained_model_upperbody/model.ckpt-104956.meta

echo 'downloading pretrained inception_v3 model for head...'
mkdir -p pretrained_model_head
wget --no-check-certificate https://www.dropbox.com/s/jhau5cbpjdd6t9p/model.ckpt-85007.data-00000-of-00001?dl=0 --output-document=pretrained_model_head/model.ckpt-85007.data-00000-of-00001
wget --no-check-certificate https://www.dropbox.com/s/hdt016n3sgikrmj/model.ckpt-85007.index?dl=0 --output-document=pretrained_model_head/model.ckpt-85007.index
wget --no-check-certificate https://www.dropbox.com/s/ov0ppq31tqipbzk/model.ckpt-85007.meta?dl=0 --output-document=pretrained_model_head/model.ckpt-85007.meta

echo 'finished.'
