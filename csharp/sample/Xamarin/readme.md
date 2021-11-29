# ONNX Runtime Xamarin Sample

The [VisionSample](VisionSample/VisionSample.sln) demonstrates the use of several [vision-related models](https://github.com/onnx/models/tree/f064171f7dd8e962a8a5b34eac8e1bcf83cebbde#vision), from the [ONNX Model Zoo collection](https://github.com/onnx/models/tree/f064171f7dd8e962a8a5b34eac8e1bcf83cebbde#onnx-model-zoo), by a [Xamarin.Forms](https://dotnet.microsoft.com/apps/xamarin/xamarin-forms) app. 

## Overview
The sample enables you to take/pick a photo on the device or use a sample image, if one is provided, to explore the following models.

### [Faster R-CNN](https://github.com/onnx/models/blob/f064171f7dd8e962a8a5b34eac8e1bcf83cebbde/vision/object_detection_segmentation/faster-rcnn)

Detects 80 different classes in an image providing detection boxes and scores for each label.

### [ResNet](https://github.com/onnx/models/tree/f064171f7dd8e962a8a5b34eac8e1bcf83cebbde/vision/classification/resnet#resnet)

Classifies the major object in the image into 1,000 pre-defined classes.

### [Ultraface](https://github.com/onnx/models/tree/f064171f7dd8e962a8a5b34eac8e1bcf83cebbde/vision/body_analysis/ultraface#ultra-lightweight-face-detection-model)

Lightweight face detection model designed for edge computing devices providing detection boxes and scores for a given image.

The sample also demonstrates how to switch between the default **CPU EP ([Execution Provider](https://onnxruntime.ai/docs/execution-providers))** and platform-specific options. In this case, [NNAPI](https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html) for Android and [Core ML](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html) for iOS.

## Getting Started

> [!IMPORTANT] Until ORT is officially released with Xamarin support in the nuget package you need to take the following additional steps:
>   - Get the managed and native nuget packages from the internal Zip-Nuget-Java packaging pipeline for a build of master
>   - Put those packages in a local directory
>   - Update the nuget.config to point to that directory
>
> There are some [known issues](#known-issues) that could impact aspects of the sample.

The sample should build and run as-is, but you must include the model(s) you wish to explore in the appropraite directory for them to appear as options. 

The [VisionSample](VisionSample/VisionSample.sln) looks for model files in a folder in this directory called **Models**. You must create this folder if you have not done so already. 

From this directory:
```
> mkdir Models
```

With **Models** set as the current directory, you can then use [wget](https://www.gnu.org/software/wget) to download the desired model files to this folder.

From this directory:
```
> cd Models
> wget <model_url>
```

> [!NOTE] 
> You may need to reload [VisionSample.csproj](VisionSample/VisionSample.csproj) before newly added model files will appear in [Visual Studio Solution Explorer](https://docs.microsoft.com/visualstudio/ide/use-solution-explorer?view=vs-2022).

### Model Downloads

| MODEL  | DOWNLOAD URL | Size   |
| ------ | ------------ | ------ |
| Faster R-CNN  | https://github.com/onnx/models/blob/f064171f7dd8e962a8a5b34eac8e1bcf83cebbde/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-10.onnx | 160.0 MB |
| ResNet  | https://github.com/onnx/models/raw/f064171f7dd8e962a8a5b34eac8e1bcf83cebbde/vision/classification/resnet/model/resnet50-v2-7.onnx | 97.7 MB |
| Ultraface  | https://github.com/onnx/models/raw/f064171f7dd8e962a8a5b34eac8e1bcf83cebbde/vision/body_analysis/ultraface/models/version-RFB-320.onnx | 1.21 MB |

## Known Issues

### Several open issues relating to Xamarin Media components

The sample leverages [Xamarin.Essetials MediaPicker APIs](https://docs.microsoft.com/xamarin/essentials/media-picker?context=xamarin%2Fxamarin-forms&tabs=android) and [Xam.Plugin.Media](https://github.com/jamesmontemagno/MediaPlugin#media-plugin-for-xamarin-and-windows) to handle taking and picking photos in a cross-platform manner. There are several open issues which may impact the ability to use these components on specific devices. 

- [Xamarin.Essentials](https://github.com/xamarin/Essentials/issues)
- [Xam.Plugin.Media](https://github.com/jamesmontemagno/MediaPlugin/issues)

The take and capture photo options are provided as a convenience but are not directly related to the use of [ONNX Runtime](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime) packages by a [Xamarin.Forms](https://dotnet.microsoft.com/apps/xamarin/xamarin-forms) app. If you're unable to use those options, you can explore use of the models using the sample image option instead.

### [MissingMethodException](https://docs.microsoft.com/dotnet/api/system.missingmethodexception) related to [ReadOnlySpan&lt;T>](https://docs.microsoft.com/dotnet/api/system.readonlyspan-1)

In [Visual Studio 2022](https://visualstudio.microsoft.com), [Hot Reload](https://docs.microsoft.com/xamarin/xamarin-forms/xaml/hot-reload) loads some additional dependencies including [System.Memory](https://www.nuget.org/packages/System.Memory) and [System.Buffers](https://www.nuget.org/packages/System.Buffers) which may cause conflicts with packages such as [ONNX Runtime](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Managed). The workaround is to [Disable Hot Reload](https://docs.microsoft.com/xamarin/xamarin-forms/xaml/hot-reload#enable-xaml-hot-reload-for-xamarinforms) until the [issue](https://developercommunity.visualstudio.com/t/bug-in-visual-studio-2022-xamarin-signalr-method-n/1528510#T-N1585809) has been addressed.