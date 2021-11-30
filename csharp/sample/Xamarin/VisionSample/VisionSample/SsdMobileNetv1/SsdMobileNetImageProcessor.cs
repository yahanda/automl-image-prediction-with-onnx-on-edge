using System;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace VisionSample
{
    public class SsdMobileNetImageProcessor : SkiaSharpImageProcessor<SsdMobileNetPrediction, byte>
    {
        protected override SKBitmap OnPreprocessSourceImage(SKBitmap sourceImage)
        {
            // Resize image
            float ratio = 800f / Math.Min(sourceImage.Width, sourceImage.Height);
            var scaledImage = sourceImage.Resize(new SKImageInfo((int)(ratio * sourceImage.Width), (int)(ratio * sourceImage.Height)), SKFilterQuality.Medium);

            return scaledImage;
        }

        protected override Tensor<byte> OnGetTensorForImage(SKBitmap image)
        {
            var bytes = image.GetPixelSpan();

            // For the raw 4-channel RGBA formatted image, the expected input length would be H x W x 4
            var expectedInputLength = image.Height * image.Width * 4;

            // For the Tensor we need 3-channel RGB format, so expected length would be H x W x 3
            var expectedOutputLength = image.Height * image.Width * 3;

            if (bytes.Length != expectedInputLength)
                throw new Exception($"The parameter {nameof(image)} is an unexpected length. Expected length is {expectedInputLength}");

            // The channelData array is expected to be in RGB order
            byte[] channelData = new byte[expectedOutputLength];

            // Extract only the desired channel data (don't want the alpha)
            var expectedChannelLength = expectedOutputLength / 3;
            var gOffset = expectedChannelLength;
            var bOffset = expectedChannelLength * 2;

            for (int i = 0, i2 = 0; i < bytes.Length; i += 4, i2++)
            {
                var r = Convert.ToByte(bytes[i]);
                var g = Convert.ToByte(bytes[i + 1]);
                var b = Convert.ToByte(bytes[i + 2]);
                channelData[i2] = r;
                channelData[i2 + gOffset] = g;
                channelData[i2 + bOffset] = b;
            }

            return new DenseTensor<byte>(channelData, new[] { 1, image.Height, image.Width, 3 });
        }

        protected override void OnApplyPrediction(SsdMobileNetPrediction prediction, SKPaint textPaint, SKPaint rectPaint, SKCanvas canvas)
        {
            var text = $"{prediction.Label}, {prediction.Score:0.00}";
            var textBounds = new SKRect();
            textPaint.MeasureText(text, ref textBounds);

            canvas.DrawRect(prediction.Box.Xmin, prediction.Box.Ymin, prediction.Box.Xmax - prediction.Box.Xmin, prediction.Box.Ymax - prediction.Box.Ymin, rectPaint);
            canvas.DrawText($"{prediction.Label}, {prediction.Score:0.00}", prediction.Box.Xmin, prediction.Box.Ymin + textBounds.Height, textPaint);
        }
    }
}