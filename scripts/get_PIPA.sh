# create data folder
mkdir -p PIPA
cd PIPA

# downloading PIPA data
wget --no-check-certificate https://people.eecs.berkeley.edu/~nzhang/datasets/annotations.zip
wget --no-check-certificate https://people.eecs.berkeley.edu/~nzhang/datasets/pipa_train.tar
wget --no-check-certificate https://people.eecs.berkeley.edu/~nzhang/datasets/pipa_val.tar
wget --no-check-certificate https://people.eecs.berkeley.edu/~nzhang/datasets/pipa_test.tar
wget --no-check-certificate https://people.eecs.berkeley.edu/~nzhang/datasets/pipa_leftover.tar

# extracting
tar -xf pipa_train.tar
tar -xf pipa_val.tar
tar -xf pipa_test.tar
tar -xf pipa_leftover.tar
unzip annotations.zip

# deleting tar & zip
rm *.zip
rm *.tar

# further clean up
rm -rf __MACOSX

