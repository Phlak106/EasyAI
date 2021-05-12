# EasyAI
A project to make working with popular pre-trained models easier.

It seems like each developer has to rewrite the boiler-plate code to 
load models, pre-process inputs, and interpret outputs of machine 
learning models. This project attempts to establish a pattern for 
publishing AI packages that reduces that start-up cost for trying out 
new models. This can be especially useful for prototyping scenarios.
This project also attempts to establish a pattern for constructing
task-specific APIs for the models e.g. object detection models take an
image and return detected objects.   

Model packages define their dependencies such as the execution runtime 
(TensorFlow, ONNX, etc.) and abstract the consumer from these details.

# See Also
https://github.com/AlturosDestinations/Alturos.Yolo - Same idea, but for only one model.