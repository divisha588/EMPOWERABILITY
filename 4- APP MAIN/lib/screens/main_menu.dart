// // import 'package:empower_ability/screens/emergency_screen.dart';
// // import 'package:empower_ability/screens/medicine_reminder.dart';
// // import 'package:empower_ability/screens/money_counter.dart';
// // import 'package:empower_ability/screens/object_detection.dart';
// // import 'package:empower_ability/screens/sign_action_detection.dart';
// // import 'package:empower_ability/screens/speech_to_text.dart';
// // import 'package:empower_ability/screens/text_to_speech.dart';
// // import 'package:empower_ability/services/notif_services.dart';
// // import 'package:flutter/material.dart';

// // class MainMenu extends StatefulWidget {
// //   const MainMenu({super.key});

// //   @override
// //   State<MainMenu> createState() => _MainMenuState();
// // }

// // class _MainMenuState extends State<MainMenu> {
// //   @override
// //   Widget build(BuildContext context) {
// //     return Scaffold(
// //       appBar: AppBar(
// //         backgroundColor: Colors.red,
// //         centerTitle: true,
// //         title: const Text(
// //           "EmpowerAbility",
// //           style: TextStyle(
// //             color: Colors.white,
// //           ),
// //         ),
// //       ),
// //       body: GridView.count(
// //         crossAxisCount: 2,
// //         children: [
// //           IconButton(
// //               onPressed: () {
// //                 Navigator.push(context,
// //                     MaterialPageRoute(builder: (context) => TextToSpeech()));
// //               },
// //               icon: Image.asset('lib/icons/text-to-speech.png')),
// //           IconButton(
// //               onPressed: () {
// //                 Navigator.push(context,
// //                     MaterialPageRoute(builder: (context) => SpeechScreen()));
// //               },
// //               icon: Image.asset('lib/icons/speech-to-text.png')),
// //           const Padding(
// //             padding: EdgeInsets.only(left: 30),
// //             child: Text(
// //               "Text to speech",
// //               style: TextStyle(
// //                 fontSize: 20,
// //               ),
// //             ),
// //           ),
// //           const Padding(
// //             padding: EdgeInsets.only(left: 40),
// //             child: Text(
// //               "Speech to text",
// //               style: TextStyle(fontSize: 20),
// //             ),
// //           ),
// //           IconButton(
// //               onPressed: () {
// //                 Navigator.push(
// //                     context,
// //                     MaterialPageRoute(
// //                         builder: (context) => MedicineReminder()));
// //               },
// //               icon: Image.asset("lib/icons/reminder.png")),
// //           IconButton(
// //               onPressed: () {
// //                 Navigator.push(context,
// //                     MaterialPageRoute(builder: (context) => EmergencyScreen()));
// //               },
// //               icon: Image.asset('lib/icons/sos.png')),
// //           const Padding(
// //             padding: EdgeInsets.only(left: 30),
// //             child: Text(
// //               "Medicine Reminder",
// //               style: TextStyle(
// //                 fontSize: 20,
// //               ),
// //             ),
// //           ),
// //           const Padding(
// //             padding: EdgeInsets.only(left: 40),
// //             child: Text(
// //               "      Emergency",
// //               style: TextStyle(fontSize: 20),
// //             ),
// //           ),
// //           IconButton(
// //               onPressed: () {
// //                 Navigator.push(
// //                     context,
// //                     MaterialPageRoute(
// //                         builder: (context) => SignActionDetection()));
// //               },
// //               icon: Image.asset('lib/icons/ok.png')),
// //           IconButton(
// //               onPressed: () {
// //                 Navigator.push(context,
// //                     MaterialPageRoute(builder: (context) => MoneyCounter()));
// //               },
// //               icon: Image.asset('lib/icons/money.png')),
// //           const Padding(
// //             padding: EdgeInsets.only(left: 40),
// //             child: Text(
// //               "Sign Action detection",
// //               style: TextStyle(fontSize: 20),
// //             ),
// //           ),
// //           const Padding(
// //             padding: EdgeInsets.only(left: 40),
// //             child: Text(
// //               "Money Counter",
// //               style: TextStyle(fontSize: 20),
// //             ),
// //           ),
// //           IconButton(
// //               onPressed: () {
// //                 Navigator.push(
// //                     context,
// //                     MaterialPageRoute(
// //                         builder: (context) => ObjectDetectionScreen()));
// //               },
// //               icon: Image.asset('lib/icons/object_detection.png'))
// //         ],
// //       ),
// //     );
// //   }
// // }
// import 'package:flutter/material.dart';
// import 'package:empower_ability/screens/emergency_screen.dart';
// import 'package:empower_ability/screens/medicine_reminder.dart';
// import 'package:empower_ability/screens/money_counter.dart';
// import 'package:empower_ability/screens/object_detection.dart';
// import 'package:empower_ability/screens/sign_action_detection.dart';
// import 'package:empower_ability/screens/speech_to_text.dart';
// import 'package:empower_ability/screens/text_to_speech.dart';

// class MainMenu extends StatelessWidget {
//   const MainMenu({Key? key}) : super(key: key);

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       appBar: AppBar(
//         backgroundColor: Colors.red,
//         centerTitle: true,
//         title: const Text(
//           "EmpowerAbility",
//           style: TextStyle(
//             color: Colors.white,
//           ),
//         ),
//       ),
//       body: GridView.count(
//         crossAxisCount: 2,
//         childAspectRatio: 1.0,
//         padding: EdgeInsets.all(16.0),
//         mainAxisSpacing: 16.0,
//         crossAxisSpacing: 16.0,
//         children: [
//           _buildMenuItem(
//             context,
//             'Text to Speech',
//             'lib/icons/text-to-speech.png',
//             TextToSpeech(),
//           ),
//           _buildMenuItem(
//             context,
//             'Speech to Text',
//             'lib/icons/speech-to-text.png',
//             SpeechScreen(),
//           ),
//           _buildMenuItem(
//             context,
//             'Medicine Reminder',
//             'lib/icons/reminder.png',
//             MedicineReminder(),
//           ),
//           _buildMenuItem(
//             context,
//             'Emergency',
//             'lib/icons/sos.png',
//             EmergencyScreen(),
//           ),
//           _buildMenuItem(
//             context,
//             'Sign Action Detection',
//             'lib/icons/ok.png',
//             SignActionDetection(),
//           ),
//           _buildMenuItem(
//             context,
//             'Money Counter',
//             'lib/icons/money.png',
//             MoneyCounter(),
//           ),
//           _buildMenuItem(
//             context,
//             'Object Detection',
//             'lib/icons/object_detection.png',
//             ObjectDetectionScreen(),
//           ),
//         ],
//       ),
//     );
//   }

//   Widget _buildMenuItem(
//       BuildContext context, String title, String iconPath, Widget screen) {
//     return InkWell(
//       onTap: () {
//         Navigator.push(
//             context, MaterialPageRoute(builder: (context) => screen));
//       },
//       child: Card(
//         elevation: 4.0,
//         shape:
//             RoundedRectangleBorder(borderRadius: BorderRadius.circular(12.0)),
//         child: Column(
//           mainAxisAlignment: MainAxisAlignment.center,
//           children: [
//             Image.asset(iconPath, width: 64.0, height: 64.0),
//             SizedBox(height: 8.0),
//             Text(
//               title,
//               style: TextStyle(
//                 fontSize: 16.0,
//                 fontWeight: FontWeight.bold,
//               ),
//               textAlign: TextAlign.center,
//             ),
//           ],
//         ),
//       ),
//     );
//   }
// }
import 'package:flutter/material.dart';
import 'package:empower_ability/screens/emergency_screen.dart';
import 'package:empower_ability/screens/medicine_reminder.dart';
import 'package:empower_ability/screens/money_counter.dart';
import 'package:empower_ability/screens/object_detection.dart';
import 'package:empower_ability/screens/sign_action_detection.dart';
import 'package:empower_ability/screens/speech_to_text.dart';
import 'package:empower_ability/screens/text_to_speech.dart';

class MainMenu extends StatelessWidget {
  const MainMenu({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.red,
        centerTitle: true,
        title: const Text(
          "EmpowerAbility",
          style: TextStyle(
            color: Colors.white,
          ),
        ),
      ),
      body: GridView.count(
        crossAxisCount: 2,
        childAspectRatio: 1.0,
        padding: EdgeInsets.all(16.0),
        mainAxisSpacing: 16.0,
        crossAxisSpacing: 16.0,
        children: [
          _buildMenuItem(
            context,
            'Text to Speech',
            'lib/icons/text-to-speech.png',
            TextToSpeech(),
          ),
          _buildMenuItem(
            context,
            'Speech to Text',
            'lib/icons/speech-to-text.png',
            SpeechScreen(),
          ),
          _buildMenuItem(
            context,
            'Medicine Reminder',
            'lib/icons/reminder.png',
            MedicineReminder(),
          ),
          _buildMenuItem(
            context,
            'Emergency',
            'lib/icons/sos.png',
            EmergencyScreen(),
          ),
          _buildMenuItem(
            context,
            'Sign Action Detection',
            'lib/icons/ok.png',
            SignActionDetection(),
          ),
          _buildMenuItem(
            context,
            'Money Counter',
            'lib/icons/money.png',
            MoneyCounter(),
          ),
          _buildMenuItem(
            context,
            'Object Detection',
            'lib/icons/object_detection.png',
            ObjectDetectionScreen(),
          ),
        ],
      ),
    );
  }

  Widget _buildMenuItem(
      BuildContext context, String title, String iconPath, Widget screen) {
    return InkWell(
      onTap: () {
        Navigator.push(
          context,
          PageRouteBuilder(
            pageBuilder: (context, animation, secondaryAnimation) => screen,
            transitionsBuilder:
                (context, animation, secondaryAnimation, child) {
              return FadeTransition(
                opacity: animation,
                child: child,
              );
            },
          ),
        );
      },
      child: Card(
        elevation: 4.0,
        shape:
            RoundedRectangleBorder(borderRadius: BorderRadius.circular(12.0)),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Image.asset(iconPath, width: 64.0, height: 64.0),
            SizedBox(height: 8.0),
            Text(
              title,
              style: TextStyle(
                fontSize: 16.0,
                fontWeight: FontWeight.bold,
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}
