import os
from shutil import copyfile
anno=open("annotations/index.txt",'r')
for l in anno:
    photoset_id,photo_id,_,_,_,_,identity_id,subset_id=l.split()
    if subset_id=='1':
        # print photoset_id,photo_id,identity_id
        src="./train/"+photoset_id+"_"+photo_id+".jpg"
        if not os.path.exists("PIPA_reorganized/"+identity_id):
            os.makedirs("PIPA_reorganized/"+identity_id)
	print "New identity:"+identity_id	
        copyfile(src,"PIPA_reorganized/"+identity_id+"/"+photoset_id+"_"+photo_id+".jpg")
    # break
    

