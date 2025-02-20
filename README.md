# automl-image-prediction-with-onnx-on-edge
Make predictions with ONNX on computer vision models from AutoML on Azure IoT Edge (ARM32 device)

## Usage
1. Connect to an edge device.
1. Clone this repository.
    ```
    git clone https://github.com/yahanda/automl-image-prediction-with-onnx-on-edge.git
    ```
1. Download `model.onnx` and `labels.json` from an AutoML training run, and put them under the folder `<cloned-repository>/automl-image-prediction-with-onnx-on-edge/app/`.
1. Build and push docker image.
    ```
    cd automl-image-prediction-with-onnx-on-edge/dockerfiles
    sudo git submodule update --init
    sudo docker build -t automlimageprediction -f Dockerfile.arm32v7 ..
    sudo docker tag automlimageprediction <your-container-registry>/automlimageprediction:<version>-arm32v7
    sudo docker push <your-container-registry>/automlimageprediction:<version>-arm32v7
    ```
1. [Deploy the module to Azure IoT Edge device](https://docs.microsoft.com/en-us/azure/iot-edge/how-to-deploy-modules-portal)


