# VGG19-classifier

This repository includes two files-

    alexnet.py - This include a simple model which can be trained with data.
                 This uses dataGenerator and flow from directory which facilitate it to handle huge data.
                 
    vgg19 - vgg19 model can be finetuned using this file.
            we can select the number of layers we are going to train and also change the number of claases to classify here.
    
    inception- This file can be used find prediction using the inception model which has lesser parameters than the vgg19 and
               gives relatively simmilar top layer prediction as that of vgg19
