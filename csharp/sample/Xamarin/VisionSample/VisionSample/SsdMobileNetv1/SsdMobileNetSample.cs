using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace VisionSample
{
    public class SsdMobileNetSample : VisionSampleBase<SsdMobileNetImageProcessor>
    {
        public const string Identifier = "SSD MobileNet";
        public const string ModelFilename = "ssd_mobilenet_v1_10.onnx";

        public SsdMobileNetSample()
            : base(Identifier, ModelFilename) { }

        protected override async Task<ImageProcessingResult> OnProcessImageAsync(byte[] image, ExecutionProviderOptions executionProvider)
        {
            using var sourceImage = await Task.Run(() => ImageProcessor.GetImageFromBytes(image, 800f)).ConfigureAwait(false);
            using var preprocessedImage = await Task.Run(() => ImageProcessor.PreprocessSourceImage(image)).ConfigureAwait(false);
            var tensor = await Task.Run(() => ImageProcessor.GetTensorForImage(preprocessedImage)).ConfigureAwait(false);
            var predictions = await Task.Run(() => GetPredictions(tensor, executionProvider, sourceImage.Width, sourceImage.Height)).ConfigureAwait(false);
            var outputImage = await Task.Run(() => ImageProcessor.ApplyPredictionsToImage(predictions, sourceImage)).ConfigureAwait(false);

            return new ImageProcessingResult(outputImage);
        }

        List<SsdMobileNetPrediction> GetPredictions(Tensor<byte> tensor, ExecutionProviderOptions executionProvider, int width, int height)
        {
            // Setup inputs and outputs
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("image_tensor:0", tensor) };

            // Run inference
            using var options = new SessionOptions();

            if (executionProvider == ExecutionProviderOptions.Platform)
                options.ApplyConfiguration(nameof(ExecutionProviderOptions.Platform));

            using var session = new InferenceSession(Model, options); // This takes a long time to create!!
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);

            // Postprocess to get predictions
            var resultsArray = results.ToArray();
            var boxes = resultsArray[0].AsEnumerable<float>().ToArray();
            var classes = resultsArray[1].AsEnumerable<float>().ToArray();
            var scores = resultsArray[2].AsEnumerable<float>().ToArray();
            var numPredictions = resultsArray[3].AsEnumerable<float>().ToArray();

            var predictions = new List<SsdMobileNetPrediction>();

            // Only first numPredications are valid
            for (int i = 0, i2 = 0; i < numPredictions[0]; i++, i2 += 4)
            {
                // The box is relative to the image size so we multiply with height and width to get pixels
                var top = boxes[i2] * height;
                var left = boxes[i2 + 1] * width;
                var bottom = boxes[i2 + 2] * height;
                var right = boxes[i2 + 3] * width;

                top = (int)Math.Max(0, Math.Floor(top + 0.5));
                left = (int)Math.Max(0, Math.Floor(left + 0.5));
                bottom = (int)Math.Min(height, Math.Floor(bottom + 0.5));
                right = (int)Math.Min(width, Math.Floor(right + 0.5));

                predictions.Add(new SsdMobileNetPrediction
                {
                    Box = new PredictionBox(left, top, right, bottom),
                    Label = SsdMobileNetLabelMap.Labels[(int)classes[i] -1],
                    Score = scores[i]
                }) ;
            }

            return predictions;
        }
    }
}