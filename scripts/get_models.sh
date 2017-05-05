mkdir -p models

echo 'downloading CPM models...'
mkdir -p models/CPM/_trained_person_MPI/
mkdir -p models/CPM/_trained_MPI/
wget --no-check-certificate https://www.dropbox.com/s/d5x49xnhjiru2af/params.pkl?dl=0 --output-document=models/CPM/_trained_person_MPI/params.pkl
wget --no-check-certificate https://www.dropbox.com/s/uvhkd1l8v65u8ed/params.pkl?dl=0 --output-document=models/CPM/_trained_MPI/params.pkl

echo 'downloading FaceNet model...'
echo '(TODO...)'

echo 'downloading resnet model for body...'
echo '(TODO...)'