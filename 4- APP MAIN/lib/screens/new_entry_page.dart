import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:dropdown_button2/dropdown_button2.dart';
import 'package:empower_ability/models/medicine_type.dart';
import 'package:empower_ability/services/notif_services.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';

import 'package:flutter_svg/flutter_svg.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:flutter_datetime_picker_plus/flutter_datetime_picker_plus.dart'
    as picker;

DateTime scheduleTime = DateTime.now();

class EntryPage extends StatefulWidget {
  const EntryPage({super.key});

  @override
  State<EntryPage> createState() => _EntryPageState();
}

class _EntryPageState extends State<EntryPage> {
  late TextEditingController nameController;
  late TextEditingController dosageController;

  late GlobalKey<ScaffoldState> _scaffoldState;

  final List<String> medicineType = ['Bottle', 'Tablet', 'Pill', 'Injection'];
  // final List<int> intervalSelection = [1, 2, 3, 4, 5];
  int selectedIntervalDuration = 0;
  String selectedMedicineType = '';
  @override
  void dispose() {
    nameController.dispose();
    dosageController.dispose();
    super.dispose();
  }

  @override
  void initState() {
    super.initState();
    nameController = TextEditingController();
    dosageController = TextEditingController();
    _scaffoldState = GlobalKey<ScaffoldState>();
  }

  addData(String medicineName, String dosage, String medicinetype,
      int interval) async {
    FirebaseFirestore.instance
        .collection('ReminderData')
        .doc(medicineName)
        .set({
      'Name': medicineName,
      'Dosage': dosage,
      'Type': medicinetype,
      'Interval': interval
    }).then((value) {
      print('Data inserted');
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text('Data Inserted')));
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      key: _scaffoldState,
      appBar: AppBar(
        centerTitle: true,
        title: Text(
          'Add new',
          style: GoogleFonts.mulish(fontSize: 20, fontWeight: FontWeight.bold),
        ),
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(15),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              PanelTitle(
                title: 'Medicine name',
                isRequired: true,
              ),
              TextFormField(
                controller: nameController,
                textCapitalization: TextCapitalization.words,
                maxLength: 12,
                decoration: const InputDecoration(
                  border: UnderlineInputBorder(),
                ),
                style: Theme.of(context).textTheme.subtitle2,
              ),
              PanelTitle(title: 'Dosage in mg', isRequired: true),
              TextFormField(
                controller: dosageController,
                keyboardType: TextInputType.number,
                maxLength: 12,
                decoration:
                    const InputDecoration(border: UnderlineInputBorder()),
                style: Theme.of(context).textTheme.subtitle2,
              ),
              PanelTitle(title: "Medicine Type", isRequired: true),
              SizedBox(
                height: 15,
              ),
              DropdownButtonFormField2(
                isExpanded: true,
                decoration: InputDecoration(
                    contentPadding: EdgeInsets.symmetric(vertical: 16),
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(5),
                    ),
                    hintText: 'Select Medicine Type',
                    hintStyle: GoogleFonts.poppins()),
                items: medicineType
                    .map((item) => DropdownMenuItem(
                          value: item,
                          child: Text(
                            item,
                            style: GoogleFonts.poppins(),
                          ),
                        ))
                    .toList(),
                onChanged: (value) {
                  selectedMedicineType = value!;
                },
                onSaved: (newValue) {
                  selectedMedicineType = newValue!;
                },
              ),
              SizedBox(
                height: 10,
              ),
              PanelTitle(title: 'Select time', isRequired: true),
              SizedBox(
                height: 15,
              ),
              TextButton(
                  onPressed: () {
                    picker.DatePicker.showDateTimePicker(
                      context,
                      showTitleActions: true,
                      onChanged: (date) => scheduleTime = date,
                      onConfirm: (date) {},
                    );
                  },
                  child: const Text('Select time')),
              SizedBox(
                height: 25,
              ),
              SizedBox(
                height: 50,
                width: double.maxFinite,
                child: ElevatedButton(
                    onPressed: () {
                      addData(
                          nameController.text.toString(),
                          dosageController.text.toString(),
                          selectedMedicineType,
                          selectedIntervalDuration);
                      NotificationService().scheduleNotification(
                        scheduledNotificationDateTime: scheduleTime,
                        title: 'Time to take medicine :)',
                      );
                      print('notification scheduled for $scheduleTime');
                    },
                    child: Text(
                      'Save',
                      style: GoogleFonts.poppins(),
                    )),
              )
            ],
          ),
        ),
      ),
    );
  }
}

class PanelTitle extends StatelessWidget {
  final String title;
  final bool isRequired;
  const PanelTitle({super.key, required this.title, required this.isRequired});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(top: 15, left: 2),
      child: Text.rich(TextSpan(children: <TextSpan>[
        TextSpan(
          text: title,
          style: GoogleFonts.poppins(fontSize: 20),
        ),
        TextSpan(
            text: isRequired ? " *" : "",
            style: GoogleFonts.poppins(fontSize: 15))
      ])),
    );
  }
}

// class GetMedicineName extends StatelessWidget {
//   final String documentId;
//   GetMedicineName({super.key, required this.documentId});

//   @override
//   Widget build(BuildContext context) {
//     CollectionReference users =
//         FirebaseFirestore.instance.collection('ReminderData');
//     return FutureBuilder(
//       future: users.doc(documentId).get(),
//       builder:
//           (BuildContext context, AsyncSnapshot<DocumentSnapshot> snapshot) {
//         if (snapshot.hasError) {
//           return Text('Something went wrong');
//         }
//         if (snapshot.hasData && !snapshot.data!.exists) {
//           return Text('Document does not exists');
//         }
//         if (snapshot.connectionState == ConnectionState.done) {
//           Map<String, dynamic> data =
//               snapshot.data!.data() as Map<String, dynamic>;
//           return Text("Medicine name: ${data['Name']}");
//         }
//         return Text('Loading');
//       },
//     );
//   }
// }
