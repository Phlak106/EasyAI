using System;

namespace EasyAI.Common
{
    /// <summary>
    /// A common interface for object detection tasks.
    /// </summary>
    public interface IObjectDetector
    {
        /// <summary>
        /// Performs the object detection.
        /// </summary>
        /// <param name="image">The image to search.</param>
        /// <param name="minConfidence">The minimum confidence level to be included in returned objects.</param>
        /// <returns>The detected objects.</returns>
        ObjectDetectorPrediction DetectObjects(byte[] image, float minConfidence);
    }
}