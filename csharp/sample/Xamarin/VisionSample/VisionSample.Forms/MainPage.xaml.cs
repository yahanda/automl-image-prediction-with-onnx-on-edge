using System;
using System.IO;
using System.Threading.Tasks;
using Plugin.Media;
using Plugin.Media.Abstractions;
using Xamarin.Essentials;
using Xamarin.Forms;

namespace VisionSample.Forms
{
    enum ImageAcquisitionMode
    {
        Sample,
        Capture,
        Pick
    }

    public partial class MainPage : ContentPage
    {
        FasterRcnnSample _fasterRcnnSample;
        FasterRcnnSample FasterRcnnSample => _fasterRcnnSample ??= new FasterRcnnSample();

        public MainPage()
        {
            InitializeComponent();

            // See:
            // ONNX Runtime Execution Providers: https://onnxruntime.ai/docs/execution-providers/
            // Core ML: https://developer.apple.com/documentation/coreml
            // NNAPI: https://developer.android.com/ndk/guides/neuralnetworks
            ExecutionProviderOptions.Items.Add(nameof(VisionSample.ExecutionProviderOptions.CPU));
            ExecutionProviderOptions.Items.Add(Device.RuntimePlatform == Device.Android ? "NNAPI" : "Core ML");
            ExecutionProviderOptions.SelectedIndex = 1;

            Samples.Items.Add(FasterRcnnSample.Name);
            Samples.SelectedIndex = 0;
        }

        async Task AcquireAndAnalyzeImageAsync(ImageAcquisitionMode acquisitionMode = ImageAcquisitionMode.Sample)
        {
            byte[] outputImage = null;

            try
            {
                SetBusyState(true);

                var imageData = acquisitionMode switch
                {
                    ImageAcquisitionMode.Capture => await TakePhotoAsync(),
                    ImageAcquisitionMode.Pick => await PickPhotoAsync(),
                    _ => await GetSampleImageAsync()
                };

                if (imageData == null)
                {
                    SetBusyState(false);
                    return;
                }

                ClearResult();

                var sessionOptionMode = ExecutionProviderOptions.SelectedItem switch
                {
                    nameof(VisionSample.ExecutionProviderOptions.CPU) => VisionSample.ExecutionProviderOptions.CPU,
                    _ => VisionSample.ExecutionProviderOptions.Platform
                };

                IVisionSample sample = Samples.SelectedItem switch
                {
                    _ => FasterRcnnSample
                };

                outputImage = await sample.ProcessImageAsync(imageData, sessionOptionMode);
            }
            finally
            {
                SetBusyState(false);
            }

            if (outputImage != null)
                ShowResult(outputImage);
        }

        Task<byte[]> GetSampleImageAsync() => Task.Run(() =>
        {
            var assembly = GetType().Assembly;

            var imageName = Samples.SelectedItem switch
            {
                _ => "demo.jpg"
            };

            using Stream stream = assembly.GetManifestResourceStream($"{assembly.GetName().Name}.SampleImages.{imageName}");
            using MemoryStream memoryStream = new MemoryStream();

            stream.CopyTo(memoryStream);
            var sampleImage = memoryStream.ToArray();

            return sampleImage;
        });

        async Task<byte[]> PickPhotoAsync()
        {
            FileResult photo;

            try
            {
                photo = await MediaPicker.PickPhotoAsync(new MediaPickerOptions { Title = "Choose photo" });
            }
            catch (FeatureNotSupportedException fnsEx)
            {
                throw new Exception("Feature is not supported on the device", fnsEx);
            }
            catch (PermissionException pEx)
            {
                throw new Exception("Permissions not granted", pEx);
            }
            catch (Exception ex)
            {
                throw new Exception($"The {nameof(PickPhotoAsync)} method threw an exception", ex);
            }

            if (photo == null)
                return null;

            var bytes = await GetBytesFromPhotoFile(photo);

            return bytes;
        }

        async Task<byte[]> TakePhotoAsync()
        {
            MediaFile photo;

            try
            {
                await CrossMedia.Current.Initialize();

                if (!CrossMedia.Current.IsCameraAvailable || !CrossMedia.Current.IsTakePhotoSupported)
                    throw new Exception("No camera available");

                photo = await CrossMedia.Current.TakePhotoAsync(new StoreCameraMediaOptions()).ConfigureAwait(false);
            }
            catch (FeatureNotSupportedException fnsEx)
            {
                throw new Exception("Feature is not supported on the device", fnsEx);
            }
            catch (PermissionException pEx)
            {
                throw new Exception("Permissions not granted", pEx);
            }
            catch (Exception ex)
            {
                throw new Exception($"The {nameof(TakePhotoAsync)} method throw an exception", ex);
            }

            if (photo == null)
                return null;

            var bytes = await GetBytesFromPhotoFile(photo);
            photo.Dispose();

            return bytes;
        }

        async Task<byte[]> GetBytesFromPhotoFile(MediaFile fileResult)
        {
            byte[] bytes;

            using Stream stream = await Task.Run(() => fileResult.GetStream());
            using MemoryStream ms = new MemoryStream();

            stream.CopyTo(ms);
            bytes = ms.ToArray();

            return bytes;
        }

        async Task<byte[]> GetBytesFromPhotoFile(FileResult fileResult)
        {
            byte[] bytes;

            using Stream stream = await fileResult.OpenReadAsync();
            using MemoryStream ms = new MemoryStream();

            stream.CopyTo(ms);
            bytes = ms.ToArray();

            return bytes;
        }

        void ClearResult()
            => MainThread.BeginInvokeOnMainThread(() => OutputImage.Source = null);

        void ShowResult(byte[] image)
            => MainThread.BeginInvokeOnMainThread(() => OutputImage.Source = ImageSource.FromStream(() => new MemoryStream(image)));

        void SetBusyState(bool busy)
        {
            ExecutionProviderOptions.IsEnabled = !busy;
            SamplePhotoButton.IsEnabled = !busy;
            PickPhotoButton.IsEnabled = !busy;
            TakePhotoButton.IsEnabled = !busy;
            BusyIndicator.IsEnabled = busy;
            BusyIndicator.IsRunning = busy;
        }

        ImageAcquisitionMode GetAcquisitionModeFromText(string tag) => tag switch
        {
            nameof(ImageAcquisitionMode.Capture) => ImageAcquisitionMode.Capture,
            nameof(ImageAcquisitionMode.Pick) => ImageAcquisitionMode.Pick,
            _ => ImageAcquisitionMode.Sample
        };

        void AcquireButton_Clicked(object sender, EventArgs e)
            => AcquireAndAnalyzeImageAsync(GetAcquisitionModeFromText((sender as Button).Text)).ContinueWith((task)
                => { if (task.IsFaulted) MainThread.BeginInvokeOnMainThread(()
                  => DisplayAlert("Error", task.Exception.Message, "OK")); });
    }
}