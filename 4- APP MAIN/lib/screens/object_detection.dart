import 'package:camera/camera.dart';
import 'package:empower_ability/main.dart';
import 'package:flutter/material.dart';
import 'package:tflite/tflite.dart';

class ObjectDetectionScreen extends StatefulWidget {
  const ObjectDetectionScreen({super.key});

  @override
  State<ObjectDetectionScreen> createState() => _ObjectDetectionScreenState();
}

class _ObjectDetectionScreenState extends State<ObjectDetectionScreen> {
  bool isWorking = false;
  String result = "";
  CameraController? cameraController;
  CameraImage? imgCamera;

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  @override
  void dispose() async {
    super.dispose();
    await Tflite.close();
    cameraController!.dispose();
  }

  initCamera() async {
    cameraController = CameraController(cameras![0], ResolutionPreset.medium);
    cameraController!.initialize().then((value) {
      if (!mounted) {
        return;
      }
      setState(() {
        cameraController!.startImageStream((imageFromStream) {
          if (!isWorking) {
            isWorking = true;
            imgCamera = imageFromStream;
            runModelOnStreamFrames();
          }
        });
      });
    });
  }

  runModelOnStreamFrames() async {
    if (imgCamera != null) {
      var recognition = await Tflite.runModelOnFrame(
          bytesList: imgCamera!.planes.map((plane) {
            return plane.bytes;
          }).toList(),
          imageHeight: imgCamera!.height,
          imageWidth: imgCamera!.width,
          imageMean: 127.5,
          imageStd: 127.5,
          rotation: 90,
          numResults: 2,
          threshold: 0.1,
          asynch: true);
      result = "";
      recognition!.forEach((response) {
        result += response['label'];
      });

      setState(() {
        result;
      });
      isWorking = false;
    }
  }

  loadModel() async {
    await Tflite.loadModel(
        model: 'assets/model_object_detection_2.tflite',
        labels: 'assets/labels_object_detection_2.txt');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Object Detection'),
        backgroundColor: Colors.red,
      ),
      body: Center(
        child: ElevatedButton(
          style: ElevatedButton.styleFrom(shape: CircleBorder(), elevation: 5),
          onPressed: () {
            initCamera();
          },
          child: Column(children: [
            Container(
              margin: EdgeInsets.only(top: 35),
              height: 400,
              width: 360,
              child: imgCamera == null
                  ? Container(
                      child: Icon(Icons.camera_front),
                    )
                  : AspectRatio(
                      aspectRatio: cameraController!.value.aspectRatio,
                      child: CameraPreview(cameraController!),
                    ),
            ),
            SizedBox(
              height: 10,
            ),
            Text(result)
          ]),
        ),
      ),
    );
  }
}
