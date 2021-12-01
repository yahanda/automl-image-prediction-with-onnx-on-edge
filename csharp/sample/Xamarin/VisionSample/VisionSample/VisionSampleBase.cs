using System;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;

namespace VisionSample
{
    public class VisionSampleBase<TImageProcessor> : IVisionSample where TImageProcessor : new()
    {
        byte[] _model;
        string _name;
        string _modelName;
        Task _initializeTask;
        TImageProcessor _imageProcessor;
        InferenceSession _cpuSession;
        InferenceSession _platformSession;

        public VisionSampleBase(string name, string modelName)
        {
            _name = name;
            _modelName = modelName;
            _ = InitializeAsync();
        }

        public string Name => _name;
        public string ModelName => _modelName;
        public byte[] Model => _model;
        public InferenceSession CpuSession => _cpuSession;
        public InferenceSession PlatformSession => _platformSession;
        public TImageProcessor ImageProcessor => _imageProcessor ??= new TImageProcessor();

        protected virtual Task<ImageProcessingResult> OnProcessImageAsync(byte[] image, ExecutionProviderOptions executionProvider) => throw new NotImplementedException();

        public Task InitializeAsync()
        {
            if (_initializeTask == null || _initializeTask.IsFaulted)
                _initializeTask = Task.Run(() => Initialize());

            return _initializeTask;
        }

        public async Task<ImageProcessingResult> ProcessImageAsync(byte[] image, ExecutionProviderOptions executionProvider)
        {
            await InitializeAsync().ConfigureAwait(false);
            return await OnProcessImageAsync(image, executionProvider);
        }

        void Initialize()
        {
            _model = ResourceLoader.GetEmbeddedResource(ModelName);
            _cpuSession = new InferenceSession(_model);
            _platformSession = new InferenceSession(_model, SessionOptionsContainer.Create(nameof(ExecutionProviderOptions.Platform)));
        }
    }
}
