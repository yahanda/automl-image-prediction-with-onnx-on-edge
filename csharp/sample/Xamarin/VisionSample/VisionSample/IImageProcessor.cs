using System.Collections.Generic;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace VisionSample
{
    public interface IImageProcessor<TImage, TPrediction>
    {
        TImage PreprocessSourceImage(byte[] sourceImage);
        Tensor<float> GetTensorForImage(TImage image);
        byte[] ApplyPredictionsToImage(IList<TPrediction> predictions, TImage image);
    }
}