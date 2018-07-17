#Google cloud:

## Install google cloud sdk:
### Create environment variable for correct distribution
export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)"

### Add the Cloud SDK distribution URI as a package source
echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

### Import the Google Cloud Platform public key
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

### Update the package list and install the Cloud SDK
sudo apt-get update && sudo apt-get install google-cloud-sdk

### Init gcloud 
gcloud init

##Mount 
gcsfuse --implicit-dirs gpu-machine-207012-mlengine /home/ndlong95/gcloud-storage

##Unmount 
fusermount -u /home/ndlong95/gcloud-storage


## Copy file:
### From computer terminal:
gcloud compute scp --recurse  /home/duclong002/pretrained_model/keras/resnet152_weights_tf.h5 k80-u16:~/
gcloud compute scp --recurse /home/duclong002/Dataset/JPEG_data k80-u16:~/


# Run
cd keras_fine_tuning/
PYTHONPATH='.' python3 keras_impl/keras_finetune.py 
