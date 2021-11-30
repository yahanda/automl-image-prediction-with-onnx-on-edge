using System;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace VisionSample
{
    public class FasterRcnnImageProcessor : SkiaSharpImageProcessor<FasterRcnnPrediction, float>
    {
        protected override SKBitmap OnPreprocessSourceImage(SKBitmap sourceImage)
        {
            // Resize image
            float ratio = 800f / Math.Min(sourceImage.Width, sourceImage.Height);
            var scaledImage = sourceImage.Resize(new SKImageInfo((int)(ratio * sourceImage.Width), (int)(ratio * sourceImage.Height)), SKFilterQuality.Medium);

            return scaledImage;
        }

        protected override Tensor<float> OnGetTensorForImage(SKBitmap image)
        {
            var paddedHeight = (int)(Math.Ceiling(image.Height / 32f) * 32f);
            var paddedWidth = (int)(Math.Ceiling(image.Width / 32f) * 32f);

            Tensor<float> input = new DenseTensor<float>(new[] { 3, paddedHeight, paddedWidth });
            var mean = new[] { 102.9801f, 115.9465f, 122.7717f };

            for (int y = paddedHeight - image.Height; y < image.Height; y++)
            {
                for (int x = paddedWidth - image.Width; x < image.Width; x++)
                {
                    var pixel = image.GetPixel(x, y);
                    input[0, y, x] = pixel.Blue - mean[0];
                    input[1, y, x] = pixel.Green - mean[1];
                    input[2, y, x] = pixel.Red - mean[2];
                }
            }

            return input;
        }

        protected override void OnApplyPrediction(FasterRcnnPrediction prediction, SKPaint textPaint, SKPaint rectPaint, SKCanvas canvas)
        {
            var text = $"{prediction.Label}, {prediction.Confidence:0.00}";
            var textBounds = new SKRect();
            textPaint.MeasureText(text, ref textBounds);

            canvas.DrawRect(prediction.Box.Xmin, prediction.Box.Ymin, prediction.Box.Xmax - prediction.Box.Xmin, prediction.Box.Ymax - prediction.Box.Ymin, rectPaint);
            canvas.DrawText($"{prediction.Label}, {prediction.Confidence:0.00}", prediction.Box.Xmin, prediction.Box.Ymin + textBounds.Height, textPaint);
        }
    }
}