import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

late List<CameraDescription> cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: ObjectDetectionScreen(),
    );
  }
}

class ObjectDetectionScreen extends StatefulWidget {
  @override
  _ObjectDetectionScreenState createState() => _ObjectDetectionScreenState();
}

class _ObjectDetectionScreenState extends State<ObjectDetectionScreen> {
  late CameraController _controller;
  Interpreter? _interpreter;
  List<Map<String, dynamic>> _results = [];
  int _inputWidth = 0;
  int _inputHeight = 0;
  XFile? _capturedImage;
  bool _isCapturing = false;
  final List<String> _labels = ["controller", "leaf", "mouse", "pen"];

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  Future<void> _initialize() async {
    await _initCamera();
    await _loadModel();
  }

  Future<void> _initCamera() async {
    try {
      _controller = CameraController(cameras[0], ResolutionPreset.medium);
      await _controller.initialize();
      if (mounted) setState(() {});
    } catch (e) {
      print('Error initializing camera: $e');
    }
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/detect.tflite');
      final inputShape = _interpreter!.getInputTensor(0).shape;
      _inputHeight = inputShape[1];
      _inputWidth = inputShape[2];
      print('Model loaded with input shape: $inputShape');
    } catch (e) {
      print('Error loading model: $e');
    }
  }

  Future<void> _captureAndProcess() async {
    if (!_controller.value.isInitialized || _isCapturing || _interpreter == null) return;

    setState(() => _isCapturing = true);

    try {
      final image = await _controller.takePicture();
      setState(() {
        _capturedImage = image;
        _results.clear();
      });
      await _processImage(File(image.path));
    } catch (e) {
      print('Error capturing or processing image: $e');
    } finally {
      setState(() => _isCapturing = false);
    }
  }

  Future<void> _processImage(File imageFile) async {
    final bytes = await imageFile.readAsBytes();
    final originalImage = img.decodeImage(bytes);

    if (originalImage == null) {
      print('Could not decode image');
      return;
    }

    final resizedImage = img.copyResize(originalImage, width: _inputWidth, height: _inputHeight);
    final input = Float32List(_inputWidth * _inputHeight * 3);
    _normalizeImage(resizedImage, input);

    final outputShape = _interpreter!.getOutputTensor(0).shape;
    print('Model output shape: $outputShape');
    final output = Float32List(outputShape.reduce((a, b) => a * b));
    final inputs = [input.buffer];
    final outputs = {0: output.buffer};

    _interpreter!.runForMultipleInputs(inputs, outputs);
    print('Raw output tensor: $output');
    _processClassificationOutput(output);
  }

  void _normalizeImage(img.Image image, Float32List outputBuffer) {
    int index = 0;
    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        final pixel = image.getPixel(x, y);
        final r = img.getRed(pixel) / 255.0;
        final g = img.getGreen(pixel) / 255.0;
        final b = img.getBlue(pixel) / 255.0;

        outputBuffer[index++] = r;
        outputBuffer[index++] = g;
        outputBuffer[index++] = b;
      }
    }
  }

  void _processClassificationOutput(Float32List output) {
    _results.clear();

    if (output.isNotEmpty) {
      int predictedClassIndex = 0;
      double maxProbability = output[0];

      for (int i = 1; i < output.length; i++) {
        if (output[i] > maxProbability) {
          maxProbability = output[i];
          predictedClassIndex = i;
        }
      }

      final predictedLabel = _labels[predictedClassIndex];
      final confidence = maxProbability;

      _results.add({
        'label': predictedLabel,
        'confidence': confidence,
      });
    }

    setState(() {});
  }

  @override
  void dispose() {
    _controller.dispose();
    _interpreter?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_controller.value.isInitialized) {
      return Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    final size = MediaQuery.of(context).size;

    return Scaffold(
      appBar: AppBar(title: Text('Capture and Classify')),
      body: Column(
        children: [
          Expanded(
            child: _capturedImage == null
                ? SizedBox(
              width: size.width,
              child: AspectRatio(
                aspectRatio: _controller.value.aspectRatio,
                child: CameraPreview(_controller),
              ),
            )
                : Stack(
              children: [
                SizedBox.expand(
                  child: Image.file(
                    File(_capturedImage!.path),
                    fit: BoxFit.contain,
                  ),
                ),
                if (_results.isNotEmpty)
                  Positioned(
                    top: 20,
                    left: 20,
                    child: Container(
                      padding: EdgeInsets.all(8),
                      color: Colors.white.withOpacity(0.8),
                      child: Text(
                        "${_results[0]['label']} (${(_results[0]['confidence'] * 100).toStringAsFixed(2)}%)",
                        style: TextStyle(
                          color: Colors.black,
                          fontWeight: FontWeight.bold,
                          fontSize: 16,
                        ),
                      ),
                    ),
                  ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: _capturedImage == null
                ? ElevatedButton(
              onPressed: _isCapturing ? null : _captureAndProcess,
              child: Text(_isCapturing ? 'Classifying...' : 'Capture and Classify'),
            )
                : Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                ElevatedButton(
                  onPressed: () {
                    setState(() {
                      _capturedImage = null;
                      _results.clear();
                    });
                  },
                  child: Text('Retake'),
                ),
                ElevatedButton(
                  onPressed: _isCapturing ? null : _captureAndProcess,
                  child: Text(_isCapturing ? 'Classifying...' : 'Re-Classify'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}