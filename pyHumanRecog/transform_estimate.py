import os
import numpy as np
from numpy import linalg as LA
import skimage.io
import skimage.transform

import dlib
import cv2
import config

def transform_estimate_top(landmark, natural_landmark):

	if landmark.shape != natural_landmark.shape:
		print "The dimension and the number of points must be the same"
		return
	
	(Mt,q) = calculate_trans_matrix(landmark, natural_landmark)		

def calculate_trans_matrix(landmark, natural_landmark):
	dimension = landmark.shape[0]
	num_point = landmark.shape[1]
	
	#calculate center
	c_landmark = np.mean(landmark, axis=1)
	c_natural_landmark = np.mean(natural_landmark, axis=1)

	c_landmark = c_landmark.reshape(dimension,1)	
	c_natural_landmark = c_natural_landmark.reshape(dimension,1)

	#align center
	landmark_align = landmark - np.tile(c_landmark,num_point)
	natural_landmark_align = natural_landmark - np.tile(c_natural_landmark,num_point)

	M = np.dot(landmark_align, natural_landmark_align.transpose())
	#compute rotate matrix
	if dimension == 2:
		Nxx = M[0][0] + M[1][1]
		Nyx = M[0][1] - M[1][0]

		N = np.array([[Nxx, Nyx],[Nyx, -Nxx]])
		D, V = LA.eig(N)

		emax = np.argmax(np.real(D))
		#gets eigenvector corresponding to maximum eigenvalue
		q = V[:,emax]
		q = np.real(q)

		q = q*np.sign(q[1]+(q[1]>=0))
		q = q/LA.norm(q)

		R11 = np.square(q[0]) - np.square(q[1])
		R21 = np.prod(q)*2

		R = np.array([[R11, -R21], [R21, R11]])
	elif dimension == 3:
		Sxx = M[0][0]
		Syx = M[1][0]
		Szx = M[2][0]

		Sxy = M[0][1]
		Syy = M[1][1]
		Szy = M[2][1]

		Sxz = M[0][2]
		Syz = M[1][2]
		Szz = M[2][2]

		N=np.array([[(Sxx+Syy+Szz), (Syz-Szy)    , (Szx-Sxz)     , (Sxy-Syx)], \
				    [(Syz-Szy)    , (Sxx-Syy-Szz), (Sxy+Syx)     , (Szx+Sxz)], \
				    [(Szx-Sxz)    , (Sxy+Syx)    , (-Sxx+Syy-Szz), (Syz+Szy)], \
				    [(Sxy-Syx)    , (Szx+Sxz)    , (Syz+Szy)     , (-Sxx-Syy+Szz)]])	
		D, V = LA.eig(N)
		emax = np.argmax(np.real(D))
		#gets eigenvector corresponding to maximum eigenvalue
		q = V[:,emax]
		q = np.real(q)

		ii = np.argmax(np.absolute(q))
		sgn = np.sign(q[ii])
		q = q*sgn

		quat = q[:]
		nrm = LA.norm(quat)

		if nrm  == 0: 
			print ("Quaternion distribution is 0")
			return 

		quat=quat/LA.norm(quat)
		q0=quat[0]
        qx=quat[1] 
        qy=quat[2] 
        qz=quat[3]
        v =np.array(quat[1:4])
        v = np.reshape(v, (3,1))

        Z=np.array([[q0,-qz,qy], \
           			[qz,q0,-qx], \
          			[-qy,qx,q0]])

        R = np.dot(v,v.transpose()) + np.dot(Z,Z)

	#compute scale
	sss = np.sum(natural_landmark_align*np.dot(R,landmark_align))/np.sum(np.square(landmark_align))

	#compute transform
	T = c_natural_landmark - np.dot(R, c_landmark*sss)

	#Mt = np.array(np.hstack((sss*R,T)), [0,0,1])
	if dimension == 2:
		Mt = np.vstack((np.hstack((sss*R,T)),[0,0,1]))
	elif dimension == 3:
		Mt = np.vstack((np.hstack((sss*R,T)),[0,0,0,1]))
		q = q/LA.norm(q)

	print Mt
	print q
	return (Mt,q)

if __name__ == '__main__':
	X = np.array([[0.272132,0.538001,0.755920,0.582317], [0.728957,0.089360,0.507490,0.100513], [0.578818,0.779569,0.136677,0.785203]])    
	s = 0.7
	R = np.array([[0.36, 0.48, -0.8], [-0.8, 0.6, 0], [0.48,0.64,0.6]]) 
	T = np.array([[45], [-78], [98]])
	Y = s*np.dot(R,X) + np.tile(T,4)

	print ("X",X)
	print ("Y",Y)	
	transform_estimate_top(X, Y)