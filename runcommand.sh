#! /bin/bash

# Optionally, set default values
# var1="default value for var1"
# var1="default value for var2"

. train_parameters.config

export CUDA_VISIBLE_DEVICES=$CudaDevices

python main.py --network_name $NetworkName --epochs $Epochs --batch_size $BatchSize --images_per_class $ImagesperClass --classes_training $ClassesTrain --data_dir $DataDir --new_lr --lr $LearningRate



