using System;
using System.Collections.Generic;

namespace EasyAI.Common
{
    public interface IPrediction
    {
        public IEnumerable<ObjectClass> TopObjectClasses(int n);
        public int NumDetectedClasses();
        public byte[] ImageWithBoundingBoxes();
    }
}