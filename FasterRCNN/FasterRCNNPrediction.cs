using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using EasyAI.Common;
using OpenCvSharp;

namespace EasyAI.Common
{
    public class FasterRCNNPrediction : IPrediction
    {
        private ConcurrentDictionary<ObjectClass, Rect> detectedObjects = new ConcurrentDictionary<ObjectClass, Rect>();
        private Mat originalImage;

        internal FasterRCNNPrediction(Mat originalImage)
        {
            this.originalImage = originalImage;
        }

        internal void AddDetectedObject(ObjectClass oc, Rect rec)
        {
            detectedObjects.TryAdd(oc, rec);
        }

        // Put boxes, labels and confidence on image
        public byte[] ImageWithBoundingBoxes()
        {
            Mat withBoundingBoxes = new Mat(originalImage);
            Parallel.ForEach(detectedObjects.Keys, p =>
            {
                Cv2.Rectangle(withBoundingBoxes, detectedObjects[p], Scalar.Red, 2);
                Cv2.PutText(withBoundingBoxes, $"{p.ClassName}, {p.Confidence:0.00}", detectedObjects[p].TopLeft, HersheyFonts.HersheyPlain, 1, Scalar.White, 1);
            });
            return withBoundingBoxes.ToBytes();
        }

        public int NumDetectedClasses() => detectedObjects.Count;

        public IEnumerable<ObjectClass> TopObjectClasses(int n) => detectedObjects.Keys.OrderByDescending(x => x.Confidence).Take(n);
    }
}