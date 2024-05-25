// TODO Implement this library.
import 'package:empower_ability/screens/add_contacts.dart';
import 'package:empower_ability/screens/contacts_page.dart';
import 'package:flutter/material.dart';
// ignore: depend_on_referenced_packages

class NavBar extends StatelessWidget {
  const NavBar({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Drawer(
      child: ListView(
        children: [
          const UserAccountsDrawerHeader(
              accountName: Text("Welcome to the Security App"),
              accountEmail: Text("Don't worry you are safe now!")),
          ListTile(
            leading: const Icon(Icons.contacts),
            title: const Text("Edit Emergency Contacts"),
            onTap: () => {
              Navigator.push(
                  context,
                  MaterialPageRoute(
                      builder: (context) => const AddContactsPage()))
            },
          ),
          const Divider(),
          // ListTile(
          //     leading: const Icon(Icons.tips_and_updates),
          //     title: const Text("How to save ourselves?"),
          //     onTap: () {
          //       // Navigator.push(context, MaterialPageRoute(
          //       //   builder: (context)=>TipsPage()
          //       // ));
          //     }),
          // const Divider(),
          // ListTile(
          //   leading: const Icon(Icons.local_police),
          //   title: const Text("Locate Police Station"),
          //   onTap: () {},
          // ),
          // ListTile(
          //   leading: const Icon(Icons.local_hospital),
          //   title: const Text("Locate Hospital"),
          //   onTap: () {},
          // ),
          // const Divider(),
          // ListTile(
          //   leading: const Icon(Icons.exit_to_app),
          //   title: const Text("Exit"),
          //   onTap: () {},
          // ),
        ],
      ),
    );
  }
}
