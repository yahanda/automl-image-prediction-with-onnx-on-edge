using System.Threading.Tasks;

namespace VisionSample
{
    public interface IVisionSample
    {
        public string Name { get; }
        public Task<byte[]> ProcessImageAsync(byte[] image, ExecutionProviderOptions executionProvider);
    }
}