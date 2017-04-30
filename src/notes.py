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



    # inputShape = _obtain_input_shape(
    #     (imageWidth, imageHeight, 3), 
    #     default_size=imageWidth,
    #     min_size=48,
    #     data_format=K.image_data_format(),
    #     include_top=False
    # )
    # imgInput = Input(shape=inputShape)
    # # Block 1
    # x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(imgInput)
    # x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # # Block 2
    # x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    # x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)


    # # Block 3
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    # x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # # Block 4
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # # Block 5
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    # x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # # Final FC layers
    # x = Flatten(name='flatten')(x)
    # x = Dense(4096, activation='relu', name='fc1')(x)
    # x = Dense(4096, activation='relu', name='fc2')(x)
    # #x = Dense(1000, activation='softmax', name='predictions')(x)
    # x = Dense(colorChannelSize, activation='tanh', name='colorize')(x)
    
    # model = Model(imgInput, x)

    # print("model:", type(model))
    # modelJson = model.to_json()
    # with open("model2.json", "w") as jsonFile:
    #     jsonFile.write(modelJson)
