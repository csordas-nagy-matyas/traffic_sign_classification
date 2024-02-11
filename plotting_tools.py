import matplotlib.pyplot as plt


def show_images(x):

    fig, ax = plt.subplots(3,10, figsize=(16, 6))
    for i, ax in enumerate(ax.flat):
        ax.imshow(x[i] / 255.0) #cmap="gray"
    #l = [ax[i].set_title(mapping[str(y[i])]) for i in range(0,10)]
    plt.show() 

def show_random_images(mapping, train_datagen):


    O = [(x, y) for x, y in train_datagen]

    fig, ax = plt.subplots(1,5, figsize=(16, 6))
    labels = [item for item in O]
    l = [ax[i].imshow(O[i][0][0]) for i in range(0,5)]
    l = [ax[i].set_title(mapping[str(O[i][1][0])]) for i in range(0,5)]
    plt.show() 


def show_predictions(model, mapping, test_imgs, test_labels, img_width, img_height, n_channels):
    fig, ax = plt.subplots(2, 4, figsize=(12, 6))

    for i, ax in enumerate(ax.flat):
        ax.imshow(test_imgs[i]) #/255.0
        pred_class = model.predict(test_imgs[i].reshape(1, img_width, img_height, n_channels)).argmax()
        pred_class = mapping[str(pred_class)]
        true_class = mapping[str(test_labels[i])]
        # print the predicted class label as the title of the image
        ax.set_title("True label: " + true_class + "\nPredicted: " + pred_class, fontsize=15)
        ax.axis("off")
    plt.show()

def show_history(history):
    #plot train and validation accuracy curves
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

   #plot train and validation loss curves
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()