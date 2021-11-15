using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace VisionSample
{
    public class SkiaSharpImageProcessor<TPrediction> : IImageProcessor<SKBitmap, TPrediction>
    {
        public byte[] ApplyPredictionsToImage(IList<TPrediction> predictions, SKBitmap image)
        {
            // Annotate image to reflect predictions and save for viewing
            using SKSurface surface = SKSurface.Create(new SKImageInfo(image.Width, image.Height));
            using SKCanvas canvas = surface.Canvas;
            using SKPaint textPaint = new SKPaint { TextSize = 32, Color = SKColors.White };
            using SKPaint rectPaint = new SKPaint { StrokeWidth = 2f, IsStroke = true, Color = SKColors.Red };

            canvas.DrawBitmap(image, 0, 0);

            foreach (var prediction in predictions)
                OnApplyPrediction(prediction, textPaint, rectPaint, canvas);

            canvas.Flush();
            using var snapshot = surface.Snapshot();
            using var imageData = snapshot.Encode(SKEncodedImageFormat.Jpeg, 100);
            byte[] bytes = imageData.ToArray();

            return bytes;
        }

        protected virtual SKBitmap OnPreprocessSourceImage(SKBitmap sourceImage) => sourceImage;
        protected virtual Tensor<float> OnGetTensorForImage(SKBitmap image) => throw new NotImplementedException();
        protected virtual void OnApplyPrediction(TPrediction prediction, SKPaint textPaint, SKPaint rectPaint, SKCanvas canvas) { }

        public Tensor<float> GetTensorForImage(SKBitmap image)
            => OnGetTensorForImage(image);

        public SKBitmap PreprocessSourceImage(byte[] sourceImage)
        {
            // Read image
            using var image = SKBitmap.Decode(sourceImage);
            var preprocessedImage = OnPreprocessSourceImage(image);

            // Handle orientation
            // See: https://github.com/mono/SkiaSharp/issues/1551#issuecomment-756685252
            using var memoryStream = new MemoryStream(sourceImage);
            using var imageData = SKData.Create(memoryStream);
            using var codec = SKCodec.Create(imageData);
            var orientation = codec.EncodedOrigin;

            return HandleOrientation(preprocessedImage, orientation);
        }

        // Address issue with orientation rotation
        // See: https://stackoverflow.com/questions/44181914/iphone-image-orientation-wrong-when-resizing-with-skiasharp
        SKBitmap HandleOrientation(SKBitmap bitmap, SKEncodedOrigin orientation)
        {
            switch (orientation)
            {
                case SKEncodedOrigin.BottomRight:

                    using (var surface = new SKCanvas(bitmap))
                    {
                        surface.RotateDegrees(180, bitmap.Width / 2, bitmap.Height / 2);
                        surface.DrawBitmap(bitmap.Copy(), 0, 0);
                    }

                    return bitmap;

                case SKEncodedOrigin.RightTop:

                    using (var rotated = new SKBitmap(bitmap.Height, bitmap.Width))
                    {
                        using (var surface = new SKCanvas(rotated))
                        {
                            surface.Translate(rotated.Width, 0);
                            surface.RotateDegrees(90);
                            surface.DrawBitmap(bitmap, 0, 0);
                        }

                        rotated.CopyTo(bitmap);
                        return bitmap;
                    }

                case SKEncodedOrigin.LeftBottom:

                    using (var rotated = new SKBitmap(bitmap.Height, bitmap.Width))
                    {
                        using (var surface = new SKCanvas(rotated))
                        {
                            surface.Translate(0, rotated.Height);
                            surface.RotateDegrees(270);
                            surface.DrawBitmap(bitmap, 0, 0);
                        }

                        rotated.CopyTo(bitmap);
                        return bitmap;
                    }

                default:
                    return bitmap;
            }
        }
    }
}