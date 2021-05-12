using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using EasyAI.Common;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace EasyAI.YoloV3
{
    public class YoloV3Detector : IObjectDetector
    {
        private InferenceSession inferenceSession;
        private List<NamedOnnxValue> MODEL_INPUTS = new List<NamedOnnxValue>() { null };

        private static int input_dimension = 416;

        public YoloV3Detector()
        {
            using (var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream("EasyAI.YoloV3.yolov3-10.onnx"))
            {
                using (MemoryStream ms = new MemoryStream())
                {
                    stream.CopyTo(ms);
                    var modelBytes = ms.ToArray();
                    inferenceSession = new InferenceSession(modelBytes);
                }
            }
        }

        public YoloV3Detector(string modelFilePath)
        {
            if (String.IsNullOrEmpty(modelFilePath))
                throw new ArgumentNullException(nameof(modelFilePath));

            inferenceSession = new InferenceSession(modelFilePath);
        }

        public ObjectDetectorPrediction DetectObjects(byte[] imageBytes, float minScore = 0.7f)
        {
            if (imageBytes == null)
                throw new ArgumentNullException(nameof(imageBytes));
            using var frame = Cv2.ImDecode(imageBytes, ImreadModes.AnyColor);
            var tensor = Preprocess(frame);
            return Inference(tensor, frame, minScore);
        }

        /// <summary>
        /// Preprocess the input image according to https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov3
        /// </summary>
        private Tensor<float> Preprocess(Mat frame)
        {
            double ratio = Math.Min((float)input_dimension / frame.Width, (float)input_dimension / frame.Height);
            Cv2.Resize(frame, frame, new OpenCvSharp.Size((ratio * frame.Width), ratio * frame.Height));
            Cv2.CopyMakeBorder(frame, frame,
                (input_dimension - frame.Height) / 2,
                ((input_dimension+1) - frame.Height) / 2,
                (input_dimension - frame.Width) / 2,
                ((input_dimension+1) - frame.Width) / 2,
                BorderTypes.Constant, new Scalar(0, 0, 0));

            var inputArr = new float[3, frame.Height, frame.Width];
            Parallel.For(0, frame.Height, (y, s) =>
            {
                for (int x = 0; x < frame.Width; ++x)
                {
                    var pixel = frame.At<Vec3b>(y, x);
                    inputArr[0, y, x] = (float)pixel[0];
                    inputArr[1, y, x] = (float)pixel[1];
                    inputArr[2, y, x] = (float)pixel[2];
                }
            });
            return inputArr.ToTensor();
        }

        private ObjectDetectorPrediction Inference(Tensor<float> input, Mat frame, float minScore = 0.7f)
        {
            if(MODEL_INPUTS[0] == null)
            {
                MODEL_INPUTS[0] = NamedOnnxValue.CreateFromTensor("image", input);
            }
            else
                MODEL_INPUTS[0].Value = input;

            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = inferenceSession.Run(MODEL_INPUTS);
            // Postprocess to get predictions
            var resultsArray = results.ToArray();
            var boxes = resultsArray[0].AsTensor<float>();
            var labels = resultsArray[1].AsTensor<long>();
            var confidences = resultsArray[2].AsTensor<float>();
            var prediction = new ObjectDetectorPrediction(frame.Clone());
            Parallel.For(0, boxes.Length / 4, (i, s) =>
            {
                var idx = (int)i; // WHY COMPILER, WHY?
                if (confidences[idx] >= minScore)
                {
                    var j = idx * 4;
                    prediction.AddDetectedObject(
                        new ObjectClass(LabelMap.Labels[labels[idx]], confidences[idx]),
                        new Rect((int)boxes[j], (int)boxes[j + 1], (int)boxes[j + 2], (int)boxes[j + 3])
                    );
                }
            });

            return prediction;
        }
    }
}
