using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace VisionSample
{
    // See: https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/faster-rcnn#model
    // Model download: https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-10.onnx
    public class FasterRcnnSample : VisionSampleBase<FasterRcnnImageProcessor>
    {
        public const string Identifier = "Faster R-CNN";
        public const string ModelFilename = "faster_rcnn.onnx";

        public FasterRcnnSample()
            : base(Identifier, ModelFilename) {}

        protected override async Task<ImageProcessingResult> OnProcessImageAsync(byte[] sourceImage, ExecutionProviderOptions sessionOptionMode = ExecutionProviderOptions.CPU)
        {
            using var preprocessedImage = await Task.Run(() => ImageProcessor.PreprocessSourceImage(sourceImage)).ConfigureAwait(false);
            var tensor = await Task.Run(() => ImageProcessor.GetTensorForImage(preprocessedImage)).ConfigureAwait(false);
            var predictions = await Task.Run(() => GetPredictions(tensor, sessionOptionMode)).ConfigureAwait(false);
            var outputImage = await Task.Run(() => ImageProcessor.ApplyPredictionsToImage(predictions, preprocessedImage)).ConfigureAwait(false);

            return new ImageProcessingResult(outputImage);
        }

        List<FasterRcnnPrediction> GetPredictions(Tensor<float> input, ExecutionProviderOptions sessionOptionsMode = ExecutionProviderOptions.CPU)
        {
            // Setup inputs and outputs
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("image", input) };

            // Run inference
            using var options = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };

            if (sessionOptionsMode == ExecutionProviderOptions.Platform)
                options.ApplyConfiguration(nameof(ExecutionProviderOptions.Platform));

            using var session = new InferenceSession(Model, options);
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);

            // Postprocess to get predictions
            var resultsArray = results.ToArray();
            float[] boxes = resultsArray[0].AsEnumerable<float>().ToArray();
            long[] labels = resultsArray[1].AsEnumerable<long>().ToArray();
            float[] confidences = resultsArray[2].AsEnumerable<float>().ToArray();
            var predictions = new List<FasterRcnnPrediction>();
            var minConfidence = 0.7f;

            for (int i = 0; i < boxes.Length - 4; i += 4)
            {
                var index = i / 4;

                if (confidences[index] >= minConfidence)
                {
                    predictions.Add(new FasterRcnnPrediction
                    {
                        Box = new PredictionBox(boxes[i], boxes[i + 1], boxes[i + 2], boxes[i + 3]),
                        Label = FasterRcnnLabelMap.Labels[labels[index]],
                        Confidence = confidences[index]
                    });
                }
            }

            return predictions;
        }
    }
}