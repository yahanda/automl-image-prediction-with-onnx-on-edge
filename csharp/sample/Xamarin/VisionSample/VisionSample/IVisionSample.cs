using System.Threading.Tasks;

namespace VisionSample
{
    public interface IVisionSample
    {
        string Name { get; }
        Task InitializeAsync();
        Task<ImageProcessingResult> ProcessImageAsync(byte[] image, ExecutionProviderOptions executionProvider);
    }
}