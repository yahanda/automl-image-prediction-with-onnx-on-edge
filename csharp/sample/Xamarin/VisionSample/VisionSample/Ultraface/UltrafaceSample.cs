using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace VisionSample
{
    // See: https://github.com/onnx/models/tree/master/vision/body_analysis/ultraface#model
    // Model download: https://github.com/onnx/models/blob/master/vision/body_analysis/ultraface/models/version-RFB-320.onnx
    public class UltrafaceSample : IVisionSample
    {
        public const string Identifier = "Ultraface";

        byte[] _model;
        Task _initializeTask;
        UltrafaceImageProcessor _ultrafaceImageProcessor;

        UltrafaceImageProcessor UltrafaceImageProcessor => _ultrafaceImageProcessor ??= new UltrafaceImageProcessor();

        public UltrafaceSample() => _ = InitializeAsync();

        public string Name => Identifier;

        public Task InitializeAsync()
        {
            if (_initializeTask == null || _initializeTask.IsFaulted)
                _initializeTask = Task.Run(() => Initialize());

            return _initializeTask;
        }

        public async Task<ImageProcessingResult> ProcessImageAsync(byte[] image, ExecutionProviderOptions executionProvider)
        {
            await InitializeAsync().ConfigureAwait(false);

            using var sourceImage = await Task.Run(() => UltrafaceImageProcessor.GetImageFromBytes(image, 800f)).ConfigureAwait(false);
            using var preprocessedImage = await Task.Run(() => UltrafaceImageProcessor.PreprocessSourceImage(image)).ConfigureAwait(false);
            var tensor = await Task.Run(() => UltrafaceImageProcessor.GetTensorForImage(preprocessedImage)).ConfigureAwait(false);
            var predictions = await Task.Run(() => GetPredictions(tensor, sourceImage.Width, sourceImage.Height, executionProvider)).ConfigureAwait(false);
            var outputImage = await Task.Run(() => UltrafaceImageProcessor.ApplyPredictionsToImage(predictions, sourceImage)).ConfigureAwait(false);

            return new ImageProcessingResult(outputImage);
        }

        List<UltrafacePrediction> GetPredictions(Tensor<float> input, int sourceImageWidth, int sourceImageHeight, ExecutionProviderOptions executionProvider = ExecutionProviderOptions.CPU)
        {
            // Setup inputs and outputs
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input", input) };

            // Run inference
            using var options = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };

            if (executionProvider == ExecutionProviderOptions.Platform)
                options.ApplyConfiguration(nameof(ExecutionProviderOptions.Platform));

            using var session = new InferenceSession(_model, options);
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);

            // Postprocess
            var resultsArray = results.ToArray();
            float[] confidences = resultsArray[0].AsEnumerable<float>().ToArray();
            float[] boxes = resultsArray[1].AsEnumerable<float>().ToArray();

            // Confidences are represented by 2 values - the second is for the face
            var scores = confidences.Where((val, index) => index % 2 == 1).ToList();

            if (!scores.Any(i => i < 0.5))
                return new List<UltrafacePrediction>(); ;

            // find the best score
            float highestScore = scores.Max();
            var indexForHighestScore = scores.IndexOf(highestScore);
            var boxOffset = indexForHighestScore * 4;

            return new List<UltrafacePrediction> { new UltrafacePrediction
            {
                Confidence = scores[indexForHighestScore],
                Box = new PredictionBox(
                    boxes[boxOffset + 0] * sourceImageWidth,
                    boxes[boxOffset + 1] * sourceImageHeight,
                    boxes[boxOffset + 2] * sourceImageWidth,
                    boxes[boxOffset + 3] * sourceImageHeight)
            }};
        }

        void Initialize()
        {
            var assembly = GetType().Assembly;

            using Stream stream = assembly.GetManifestResourceStream($"{assembly.GetName().Name}.Ultraface.ultraface.onnx");
            using MemoryStream memoryStream = new MemoryStream();

            stream.CopyTo(memoryStream);
            _model = memoryStream.ToArray();
        }
    }
}