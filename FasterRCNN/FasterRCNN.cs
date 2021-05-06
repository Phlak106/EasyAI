using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using EasyAI.Common;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace EasyAI.FasterRCNN
{
    public class FasterRCNN
    {
        private const String MODEL_FILE = @"FasterRCNN.onnx";
        private static float[] MEAN = { 102.9801f, 115.9465f, 122.7717f };
        private InferenceSession inferenceSession;
        private List<NamedOnnxValue> MODEL_INPUTS = new List<NamedOnnxValue>() { null };

        public FasterRCNN() : this(MODEL_FILE) { }

        public FasterRCNN(string modelFilePath)
        {
            if (String.IsNullOrEmpty(modelFilePath))
                throw new ArgumentNullException(nameof(modelFilePath));

            inferenceSession = new InferenceSession(modelFilePath);
        }

        /// <summary>
        /// Preprocess the input image according to https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/faster-rcnn
        /// </summary>
        private void Preprocess(Mat frame)
        {
            double ratio = Math.Min(800.0 / frame.Width, 800.0 / frame.Height);
            Cv2.Resize(frame, frame, new OpenCvSharp.Size((ratio * frame.Width), ratio * frame.Height));
            Cv2.CopyMakeBorder(frame, frame, 
                (800 - frame.Height) / 2, 
                (801 - frame.Height) / 2, 
                (800 - frame.Width) / 2, 
                (801 - frame.Width) / 2, 
                BorderTypes.Constant, new Scalar(0, 0, 0));

            var inputArr = new float[3, frame.Height, frame.Width];
            Parallel.For(0, frame.Height, (y, s) =>
            {
                for( int x = 0; x < frame.Width; ++x ) {
                    var pixel = frame.At<Vec3b>(y, x);
                    inputArr[0, y, x] = (float)pixel[0];
                    inputArr[1, y, x] = (float)pixel[1];
                    inputArr[2, y, x] = (float)pixel[2];
                }
            });
        }

        private IPrediction Inference(Mat frame, float minScore)
        {
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = inferenceSession.Run(MODEL_INPUTS);
            // Postprocess to get predictions
            var resultsArray = results.ToArray();
            var boxes = resultsArray[0].AsTensor<float>();
            var labels = resultsArray[1].AsTensor<int>();
            var confidences = resultsArray[2].AsTensor<float>();
            var prediction = new FasterRCNNPrediction(frame);
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
