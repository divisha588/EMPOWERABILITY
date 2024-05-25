import 'package:empower_ability/db/db_services.dart';
import 'package:empower_ability/models/contacts.dart';
import 'package:empower_ability/screens/navbar.dart';
import 'package:flutter/material.dart';
import 'package:fluttertoast/fluttertoast.dart';
import 'package:geocoding/geocoding.dart';
import 'package:geolocator/geolocator.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:telephony/telephony.dart';

class EmergencyScreen extends StatefulWidget {
  const EmergencyScreen({super.key});

  @override
  State<EmergencyScreen> createState() => _EmergencyScreenState();
}

String createLocationLink(double lat, double lon) {
  String link = "https://www.google.com/maps/search/?api=1&query=$lat%2C$lon";
  return link;
}

void SMS(String message) async {
  final Telephony telephony = Telephony.instance;
  List<TContact> savedcontacts = await DataBaseHelper().getContactList();
  List numbers = [];
  if (savedcontacts.isEmpty) {
    Fluttertoast.showToast(msg: "No contacts");
  } else {
    Fluttertoast.showToast(msg: "Sending SOS...");
    bool? permissionsGranted = await telephony.requestPhoneAndSmsPermissions;
    if (permissionsGranted == true) {
      for (var nums in savedcontacts) {
        numbers.add(nums.number.substring(nums.number.length - 10));
      }
      print(numbers);
      for (var rec in numbers) {
        await telephony.sendSms(to: rec, message: message);
      }
    }
  }
}

class _EmergencyScreenState extends State<EmergencyScreen> {
  String location = 'Link will appear here';
  String Address = 'Address will appear here';
  bool isLoading = true;
  Position? position;

  // fetchPosition() async {
  //   bool serviceEnabled;
  //   LocationPermission permission;
  //   serviceEnabled = await Geolocator.isLocationServiceEnabled();
  //   if (!serviceEnabled) {
  //     Fluttertoast.showToast(msg: 'Location service is disabled');
  //   }
  //   permission = await Geolocator.checkPermission();
  //   if (permission == LocationPermission.denied) {
  //     permission = await Geolocator.requestPermission();
  //     if (permission == LocationPermission.denied) {
  //       Fluttertoast.showToast(msg: 'Location Permission denied');
  //     }
  //   }
  //   if (permission == LocationPermission.deniedForever) {
  //     Fluttertoast.showToast(msg: 'Location Permission denied forever');
  //   }
  //   Position currentPosition = await Geolocator.getCurrentPosition();
  //   setState(() {
  //     position = currentPosition;
  //   });
  // }

  fetchPosition() async {
    bool serviceEnabled;
    LocationPermission permission;
    serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) {
      Fluttertoast.showToast(msg: 'Location service is disabled');
      return null; // Exit the function early and return null
    }
    permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) {
        Fluttertoast.showToast(msg: 'Location Permission denied');
        return null; // Exit the function early and return null
      }
    }
    if (permission == LocationPermission.deniedForever) {
      Fluttertoast.showToast(msg: 'Location Permission denied forever');
      return null; // Exit the function early and return null
    }
    Position? currentPosition = await Geolocator.getCurrentPosition();
    if (currentPosition == null) {
      Fluttertoast.showToast(msg: 'Failed to fetch current position');
      return null; // Exit the function early and return null
    }
    setState(() {
      position = currentPosition;
    });
    return currentPosition; // Return the fetched position
  }

  Future<void> GetAddressFromLatLong(Position position) async {
    List<Placemark> placemarks =
        await placemarkFromCoordinates(position.latitude, position.longitude);
    print(placemarks);
    // Placemark place = placemarks[0];
    // Address =
    //     '${place.street}, ${place.subLocality}, ${place.locality}, ${place.postalCode}, ${place.country}';
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      endDrawer: const NavBar(),
      appBar: AppBar(
        title: Text(
          'Emergency',
          style: GoogleFonts.poppins(),
        ),
        backgroundColor: Colors.red,
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: () async {
                Position? currentPosition = await fetchPosition();
                if (currentPosition != null) {
                  location = createLocationLink(
                      currentPosition.latitude, currentPosition.longitude);
                  GetAddressFromLatLong(currentPosition);
                  String msg = "I need URGENT HELP!! $location";
                  SMS(msg);
                }
              },
              child: Text(
                'Send SOS',
                style: TextStyle(color: Colors.white, fontSize: 24),
              ),
              style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.red,
                  elevation: 5,
                  shape: CircleBorder(),
                  padding: EdgeInsets.all(100)),
            )
          ],
        ),
      ),
    );
  }
}
