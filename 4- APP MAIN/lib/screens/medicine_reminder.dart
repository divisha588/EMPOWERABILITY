import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:empower_ability/screens/new_entry_page.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class MedicineReminder extends StatefulWidget {
  const MedicineReminder({super.key});

  @override
  State<MedicineReminder> createState() => _MedicineReminderState();
}

class _MedicineReminderState extends State<MedicineReminder> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
            // title: Text("Medicine Reminder"),
            // backgroundColor: Colors.red,
            ),
        body: const Padding(
          padding: EdgeInsets.all(20),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.start,
            children: [
              TopContainer(),
              SizedBox(
                height: 20,
              ),
              Flexible(child: BottomContainer()),
            ],
          ),
        ),
        floatingActionButton: FloatingActionButton(
          onPressed: () {
            Navigator.push(
                context, MaterialPageRoute(builder: (context) => EntryPage()));
          },
          child: SizedBox(
            child: Card(
              child: Icon(
                Icons.add_outlined,
                size: 50,
                color: Colors.blueAccent,
              ),
            ),
          ),
        ));
  }
}

class TopContainer extends StatelessWidget {
  const TopContainer({super.key});

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      child: Column(
        children: [
          Container(
            alignment: Alignment.topLeft,
            padding: EdgeInsets.only(bottom: 15, top: 15),
            child: Text(
              'Worry less. \nLive healthier.',
              style: GoogleFonts.aBeeZee(
                  fontSize: 30, fontWeight: FontWeight.bold),
            ),
          ),
          SizedBox(
            height: 20,
          ),
        ],
      ),
    );
  }
}

class BottomContainer extends StatelessWidget {
  const BottomContainer({Key? key});

  @override
  Widget build(BuildContext context) {
    return StreamBuilder(
      stream: FirebaseFirestore.instance.collection('ReminderData').snapshots(),
      builder: (context, AsyncSnapshot<QuerySnapshot> snapshot) {
        if (snapshot.connectionState == ConnectionState.active) {
          if (snapshot.hasData) {
            return ListView.builder(
              itemCount: snapshot.data!.docs.length,
              itemBuilder: (context, index) {
                return Dismissible(
                  key: Key(snapshot.data!.docs[index].id),
                  direction: DismissDirection.endToStart,
                  background: Container(
                    color: Colors.red,
                    alignment: Alignment.centerRight,
                    padding: EdgeInsets.symmetric(horizontal: 20),
                    child: Icon(
                      Icons.delete,
                      color: Colors.white,
                    ),
                  ),
                  onDismissed: (direction) {
                    // Delete the item from Firestore
                    FirebaseFirestore.instance
                        .collection('ReminderData')
                        .doc(snapshot.data!.docs[index].id)
                        .delete();

                    ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(content: Text('Medicine Deleted')));
                  },
                  child: ListTile(
                    leading: CircleAvatar(
                      child: Text("${index + 1}"),
                    ),
                    title: Text("${snapshot.data!.docs[index]['Name']}"),
                    subtitle: Text("${snapshot.data!.docs[index]['Dosage']}"),
                    trailing: Text("${snapshot.data!.docs[index]["Type"]}"),
                  ),
                );
              },
            );
          } else if (snapshot.hasError) {
            return Center(
              child: Text("${snapshot.error.toString()}"),
            );
          } else {
            return Center(
              child: Text(
                'No Medicine',
                style: GoogleFonts.aBeeZee(color: Colors.red, fontSize: 15),
              ),
            );
          }
        }
        return Center(
          child: Text(
            'No Medicine',
            style: GoogleFonts.aBeeZee(color: Colors.red, fontSize: 15),
          ),
        );
      },
    );
  }
}
