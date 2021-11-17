using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace VisionSample
{
    // See: https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/faster-rcnn#model
    // Model download: https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-10.onnx
    public class FasterRcnnSample : IVisionSample
    {
        public const string Identifier = "Faster R-CNN";

        byte[] _model;
        Task _initializeTask;
        FasterRcnnImageProcessor _fasterRcnnImageProcessor;

        FasterRcnnImageProcessor FasterRcnnImageProcessor => _fasterRcnnImageProcessor ??= new FasterRcnnImageProcessor();

        public string Name => Identifier;

        public FasterRcnnSample() => _ = InitializeAsync();

        public async Task<ImageProcessingResult> ProcessImageAsync(byte[] sourceImage, ExecutionProviderOptions sessionOptionMode = ExecutionProviderOptions.CPU)
        {
            await InitializeAsync().ConfigureAwait(false);
            using var preprocessedImage = await Task.Run(() => FasterRcnnImageProcessor.PreprocessSourceImage(sourceImage)).ConfigureAwait(false);
            var tensor = await Task.Run(() => FasterRcnnImageProcessor.GetTensorForImage(preprocessedImage)).ConfigureAwait(false);
            var predictions = await Task.Run(() => GetPredictions(tensor, sessionOptionMode)).ConfigureAwait(false);
            var outputImage = await Task.Run(() => FasterRcnnImageProcessor.ApplyPredictionsToImage(predictions, preprocessedImage)).ConfigureAwait(false);

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

            using var session = new InferenceSession(_model, options);
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

        public Task InitializeAsync()
        {
            if (_initializeTask == null || _initializeTask.IsFaulted)
                _initializeTask = Task.Run(() => Initialize());

            return _initializeTask;
        }

        void Initialize()
        {
            var assembly = GetType().Assembly;

            using Stream stream = assembly.GetManifestResourceStream($"{assembly.GetName().Name}.FasterRcnn.faster_rcnn.onnx");
            using MemoryStream memoryStream = new MemoryStream();

            stream.CopyTo(memoryStream);
            _model = memoryStream.ToArray();
        }
    }
}