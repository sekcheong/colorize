width = img.shape[0]
	height = img.shape[1]
	ratio = width / height

	print ('ratio:', ratio)



	img_out = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

	print ('img_out:', img_out.shape)

	Y = img_out[:,:,0]
	U = img_out[:,:,1]
	V = img_out[:,:,2]

	print ('Y:', Y.shape)

	#get the gray component
	G = np.zeros(img.shape, dtype='uint8')
	G[:, :,0] = img_out[:,:,0] 
	G[:, :,1] = img_out[:,:,0] 
	G[:, :,2] = img_out[:,:,0] 



	#reconstruct an image from training vector

    image = reconstructImage(unpreprocessImage(trainX[0]), trainY[0])
    imageShow(image)