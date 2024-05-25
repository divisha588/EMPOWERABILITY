import 'package:empower_ability/consts.dart';
import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:google_fonts/google_fonts.dart';

class TextToSpeech extends StatefulWidget {
  const TextToSpeech({super.key});

  @override
  State<TextToSpeech> createState() => _TextToSpeechState();
}

class _TextToSpeechState extends State<TextToSpeech> {
  FlutterTts _flutterTts = FlutterTts();
  Map? _currentVoice;
  List<Map> _voices = [];
  int? _currentWordStart;
  int? _currentWordEnd;

  var _inputText = TextEditingController();

  @override
  void initState() {
    // TODO: implement initState
    super.initState();
    initTTS();
  }

  void initTTS() {
    _flutterTts.setProgressHandler((text, start, end, word) {
      setState(() {
        _currentWordStart = start;
        _currentWordEnd = end;
      });
    });
    _flutterTts.getVoices.then((data) {
      try {
        _voices = List<Map>.from(data);

        setState(() {
          _voices =
              _voices.where((_voice) => _voice["name"].contains("en")).toList();
          _currentVoice = _voices.first;
          setVoice(_currentVoice!);
        });
      } catch (e) {
        print(e);
      }
    });
  }

  void setVoice(Map voice) {
    _flutterTts.setVoice({"name": voice["name"], "locale": voice["locale"]});
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _buildUI(),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          _flutterTts.speak(TTS_INPUT);
        },
        child: Icon(Icons.speaker),
      ),
    );
  }

  Widget _buildUI() {
    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        title: const Text(
          'Text to speech',
          style: TextStyle(color: Colors.white),
        ),
        backgroundColor: Colors.red,
      ),
      body: SafeArea(
          child: Column(
        mainAxisAlignment: MainAxisAlignment.spaceEvenly,
        mainAxisSize: MainAxisSize.max,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          _speakerSelector(),
          RichText(
            textAlign: TextAlign.center,
            text: TextSpan(
              style: const TextStyle(
                  fontWeight: FontWeight.w300,
                  fontSize: 20,
                  color: Colors.black),
              children: <TextSpan>[
                if (TTS_INPUT.isEmpty)
                  TextSpan(
                      text: "Your typed text will appear here...",
                      style: GoogleFonts.aBeeZee(fontSize: 20)),
                TextSpan(
                  text: TTS_INPUT.substring(0, _currentWordStart),
                ),
                if (_currentWordStart != null)
                  TextSpan(
                      text: TTS_INPUT.substring(
                          _currentWordStart!, _currentWordEnd),
                      style: const TextStyle(
                        color: Colors.white,
                        backgroundColor: Colors.purpleAccent,
                      )),
                if (_currentWordEnd != null)
                  TextSpan(
                    text: TTS_INPUT.substring(_currentWordEnd!),
                  ),
              ],
            ),
          ),
          TextField(
            controller: _inputText,
            decoration: InputDecoration(
                focusedBorder:
                    OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
                enabledBorder:
                    OutlineInputBorder(borderRadius: BorderRadius.circular(10)),
                hintText: "Type your text here...",
                suffixIcon: IconButton(
                  icon: const Icon(Icons.send),
                  onPressed: () {
                    TTS_INPUT = _inputText.text.toString();
                    setState(() {
                      _inputText.clear();
                    });
                  },
                )),
          )
        ],
      )),
    );
  }

  Widget _speakerSelector() {
    return DropdownButton(
        value: _currentVoice,
        items: _voices
            .map((_voice) =>
                DropdownMenuItem(value: _voice, child: Text(_voice["name"])))
            .toList(),
        onChanged: (value) {});
  }
}
