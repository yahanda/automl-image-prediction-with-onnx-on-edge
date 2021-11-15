using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace VisionSample
{
    // See: https://github.com/onnx/models/tree/master/vision/classification/resnet#resnet
    public class ResNetSample : IVisionSample
    {
        public const string Identifier = "ResNet50 v2";

        byte[] _model;
        Task _initializeTask;
        ResNetImageProcessor _resNetImageProcessor;

        ResNetImageProcessor ResNetImageProcessor => _resNetImageProcessor ??= new ResNetImageProcessor();

        public ResNetSample() => _ = InitializeAsync();

        public string Name => Identifier;

        public async Task<ImageProcessingResult> ProcessImageAsync(byte[] image, ExecutionProviderOptions executionProvider)
        {
            await InitializeAsync().ConfigureAwait(false);
            using var preprocessedImage = await Task.Run(() => ResNetImageProcessor.PreprocessSourceImage(image)).ConfigureAwait(false);
            var tensor = await Task.Run(() => ResNetImageProcessor.GetTensorForImage(preprocessedImage)).ConfigureAwait(false);
            var predictions = await Task.Run(() => GetPredictions(tensor, executionProvider)).ConfigureAwait(false);
            var preprocessedImageData = await Task.Run(() => ResNetImageProcessor.GetBytesForBitmap(preprocessedImage)).ConfigureAwait(false);

            var caption = string.Empty;

            if (predictions.Any())
            {
                var builder = new StringBuilder();

                if (predictions.Any())
                    builder.Append($"Top {predictions.Count} predictions: {Environment.NewLine}{Environment.NewLine}");

                foreach (var prediction in predictions)
                    builder.Append($"{prediction.Label} ({prediction.Confidence}){Environment.NewLine}");

                caption = builder.ToString();
            }

            return new ImageProcessingResult(preprocessedImageData, caption);
        }

        List<ResNetPrediction> GetPredictions(Tensor<float> input, ExecutionProviderOptions executionProvider = ExecutionProviderOptions.CPU)
        {
            // Setup inputs and outputs
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("data", input) };

            // Run inference
            using var options = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };

            if (executionProvider == ExecutionProviderOptions.Platform)
                options.ApplyConfiguration(nameof(ExecutionProviderOptions.Platform));

            using var session = new InferenceSession(_model, options);
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);

            // Postprocess to get softmax vector
            IEnumerable<float> output = results.First().AsEnumerable<float>();
            float sum = output.Sum(x => (float)Math.Exp(x));
            IEnumerable<float> softmax = output.Select(x => (float)Math.Exp(x) / sum);

            // Extract top 10 predicted classes
            IEnumerable<ResNetPrediction> top10 = softmax
                .Select((x, i) => new ResNetPrediction
                {
                    Label = ResNetLabelMap.Labels[i],
                    Confidence = x
                })
                .OrderByDescending(x => x.Confidence)
                .Take(10);

            return top10.ToList();
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

            using Stream stream = assembly.GetManifestResourceStream($"{assembly.GetName().Name}.ResNet50v2.resnet50.onnx");
            using MemoryStream memoryStream = new MemoryStream();

            stream.CopyTo(memoryStream);
            _model = memoryStream.ToArray();
        }
    }
}