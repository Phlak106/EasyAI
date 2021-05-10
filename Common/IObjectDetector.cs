using System;

namespace EasyAI.Common
{
    public interface IObjectDetector
    {
        IObjectDetectorPrediction DetectObjects(byte[] image, float minConfidence);
    }
}