using System;

namespace EasyAI.Common
{
    public class ObjectClass : Tuple<string, float>
    {
        public string ClassName => Item1;
        public float Confidence => Item2;

        public ObjectClass(string className, float confidence) : base(className, confidence) {}
    }
}