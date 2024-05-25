import 'package:camera/camera.dart';
import 'package:empower_ability/main.dart';
import 'package:flutter/material.dart';

import 'package:tflite/tflite.dart';

class MoneyCounter extends StatefulWidget {
  const MoneyCounter({Key? key}) : super(key: key);

  @override
  _MoneyCounterState createState() => _MoneyCounterState();
}

class _MoneyCounterState extends State<MoneyCounter> {
  CameraImage? cameraImage;
  CameraController? cameraController;
  String output1 = '';
  // String output2 = '';

  @override
  void initState() {
    super.initState();
    loadModel();
    initializeCamera();
  }

  void initializeCamera() async {
    final cameras = await availableCameras();
    final camera = cameras.first;
    cameraController = CameraController(camera, ResolutionPreset.high);
    await cameraController!.initialize();
    if (mounted) {
      setState(() {});
    }
    cameraController!.startImageStream((CameraImage image) {
      cameraImage = image;
      runModel();
    });
  }

  void runModel() async {
    if (cameraImage != null) {
      final List<dynamic>? results = await Tflite.runModelOnFrame(
        bytesList: cameraImage!.planes.map((plane) {
          return plane.bytes;
        }).toList(),
        imageHeight: cameraImage!.height,
        imageWidth: cameraImage!.width,
        imageMean: 127.5,
        imageStd: 127.5,
        rotation: 90,
        numResults: 2,
        threshold: 0.1,
        asynch: true,
      );
      if (results != null && results.isNotEmpty) {
        setState(() {
          output1 = results[0]['label'];
          // output2 = results[1]['label'];
        });
      }
    }
  }

  void loadModel() async {
    await Tflite.loadModel(
      model: "assets/model_indian_currency.tflite",
      labels: "assets/labels_indian_currency.txt",
    );
  }

  @override
  void dispose() {
    super.dispose();
    cameraController?.dispose();
    Tflite.close();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Money Counter'),
        backgroundColor: Colors.red,
      ),
      body: SingleChildScrollView(
        child: Column(children: [
          Padding(
            padding: EdgeInsets.all(20),
            child: Container(
              height: 300,
              width: MediaQuery.of(context).size.width,
              child: !cameraController!.value.isInitialized
                  ? Container()
                  : AspectRatio(
                      aspectRatio: cameraController!.value.aspectRatio,
                      child: CameraPreview(cameraController!),
                    ),
            ),
          ),
          Text(
            output1,
            style: TextStyle(fontWeight: FontWeight.bold, fontSize: 20),
          ),
          // Padding(
          //   padding: const EdgeInsets.all(20),
          //   child: Container(
          //     height: 300,
          //     width: MediaQuery.of(context).size.width,
          //     child: !cameraController!.value.isInitialized
          //         ? Container()
          //         : AspectRatio(
          //             aspectRatio: cameraController!.value.aspectRatio,
          //             child: CameraPreview(cameraController!),
          //           ),
          //   ),
          // ),
          // Text(
          //   output2,
          //   style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 20),
          // ),
          Text(
            'Total amount: ${output1}',
            style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 20),
          )
        ]),
      ),
    );
  }
}
