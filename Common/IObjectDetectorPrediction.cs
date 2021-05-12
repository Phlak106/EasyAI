using System;
using System.Collections.Generic;

namespace EasyAI.Common
{
    /// <summary>
    /// An interface representing the detected objects.
    /// </summary>
    public interface IObjectDetectorPrediction
    {
        /// <summary>
        /// Gets the top n detected objects.
        /// </summary>
        /// <param name="n">The number of objects to get.</param>
        /// <returns>An enumerable containing the top n detected objects.</returns>
        public IEnumerable<ObjectClass> TopObjectClasses(int n);
        
        /// <summary>
        /// Gets the total number of detected objects.
        /// </summary>
        /// <returns>The total number of detected objects.</returns>
        public int NumDetectedClasses();
        
        /// <summary>
        /// Generates an image from the original with bounding boxes around the detected objects.
        /// </summary>
        /// <returns>Bytes of image with bounding boxes around the detected objects.</returns>
        public byte[] ImageWithBoundingBoxes();
    }
}