using System.Threading.Tasks;

namespace VisionSample
{
    public interface IVisionSample
    {
        string Name { get; }
        string ModelName { get; }
        Task InitializeAsync();
        Task UpdateExecutionProvider(ExecutionProviderOptions executionProvider);
        Task<ImageProcessingResult> ProcessImageAsync(byte[] image);
    }
}
