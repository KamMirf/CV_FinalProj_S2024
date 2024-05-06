from roboflow import Roboflow
rf = Roboflow(api_key = 'a9VkdJpM9YHFBOsfIzbB', model_format="yolov5", notebook="roboflow-yolov5")
rf = Roboflow(api_key="a9VkdJpM9YHFBOsfIzbB")
project = rf.workspace("karel-cornelis-q2qqg").project("aicook-lcv4d")
version = project.version(4)
dataset = version.download("yolov5")

#Just run this file and it will make a new folder called aicook-4 in root with all the data