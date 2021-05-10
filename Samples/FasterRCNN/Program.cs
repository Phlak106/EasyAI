using System;
using System.IO;
using System.Linq;
using EasyAI.FasterRCNN;

namespace FasterRCNN
{
    class Program
    {
        static void Main(string[] args)
        {
            string imageFilePath = args[0];
            string outImageFilePath = args[1];

            var model = new FasterRCNNDetector();
            var image = File.ReadAllBytes(imageFilePath);
            var result = model.DetectObjects(image);
            File.WriteAllBytes(outImageFilePath, result.ImageWithBoundingBoxes());
            Console.WriteLine($"Detected {result.NumDetectedClasses()} objects!");
            var topObject = result.TopObjectClasses(1).First();
            Console.WriteLine($"Most confident object is {topObject.ClassName} with confidence {topObject.Confidence}");
        }
    }
}
