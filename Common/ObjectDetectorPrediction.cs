using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using OpenCvSharp;

namespace EasyAI.Common
{
    /// <summary>
    /// An class representing the detected objects.
    /// </summary>
    public class ObjectDetectorPrediction: IDisposable
    {
        private ConcurrentDictionary<ObjectClass, Rect> detectedObjects = new ConcurrentDictionary<ObjectClass, Rect>();
        private Mat originalImage;

        public ObjectDetectorPrediction(Mat originalImage)
        {
            this.originalImage = originalImage;
        }

        public void AddDetectedObject(ObjectClass oc, Rect rec)
        {
            detectedObjects.TryAdd(oc, rec);
        }

        /// <summary>
        /// Generates an image from the original with bounding boxes around the detected objects.
        /// </summary>
        /// <returns>Bytes of image with bounding boxes around the detected objects.</returns>
        public byte[] ImageWithBoundingBoxes()
        {
            using (Mat withBoundingBoxes = originalImage.Clone()) {
                Parallel.ForEach(detectedObjects.Keys, p =>
                {
                    Cv2.Rectangle(withBoundingBoxes, detectedObjects[p], Scalar.Red, 2);
                    Cv2.PutText(withBoundingBoxes, $"{p.ClassName}, {p.Confidence:0.00}", detectedObjects[p].TopLeft, HersheyFonts.HersheyPlain, 1, Scalar.White, 1);
                });
                return withBoundingBoxes.ToBytes();
            }
        }

        /// <summary>
        /// Gets the total number of detected objects.
        /// </summary>
        /// <returns>The total number of detected objects.</returns>
        public int NumDetectedClasses() => detectedObjects.Count;

        /// <summary>
        /// Gets the top n detected objects.
        /// </summary>
        /// <param name="n">The number of objects to get.</param>
        /// <returns>An enumerable containing the top n detected objects.</returns>
        public IEnumerable<ObjectClass> TopObjectClasses(int n)
        {
            if (n < 0) throw new ArgumentOutOfRangeException(nameof(n), "Requested number of classes must be non-negative.");
            if (n > NumDetectedClasses()) throw new ArgumentOutOfRangeException(nameof(n), "Requested number of classes must not excess the number of detected classes.");
            
            return detectedObjects.Keys.OrderByDescending(x => x.Confidence).Take(n);
        }

        public void Dispose() 
        {
            originalImage.Dispose();
        }
    }
}