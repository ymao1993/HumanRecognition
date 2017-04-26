#dlib landmarks path
#dlibFacePredictor = "/home/ubuntu/tools/openface/models/dlib/shape_predictor_68_face_landmarks.dat"
#dlibFacePredictor = "/home/schen/large_scale/tool/openface/models/dlib/shape_predictor_68_face_landmarks.dat"
dlibFacePredictor = "pyHumanRecog/shape_predictor_68_face_landmarks.dat"
#The edge length in pixels of the square the image is resized to
imgDim = 100

#save head image
save_head_image = True

#save aligned head image
save_aligned_head_image = True

#save head image path
head_path = "PIPA"

#dense_output for coco similarity
dense_output = True

#normalize parameter for calculate similarity
beta0 = 1.0
beta1 = 1.0

#weighted of each region
w0 = 1.0/3
w1 = 1.0/3
w2 = 1.0/3