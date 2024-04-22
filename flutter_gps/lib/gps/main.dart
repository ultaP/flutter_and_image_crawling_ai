// main.dart
import 'package:flutter/material.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:webview_flutter/webview_flutter.dart';

import 'location_service.dart';
import 'notification_service.dart';
import 'webview_main_controller.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final LocationService locationService = LocationService(
      targetLatitude: 37.5578, targetLongitude: 127.0093, targetDistance: 50);
  final NotificationService notificationService = NotificationService();

  bool isDistanceMeasuring = false;
  var buttonText = const Text('시작');
  var loginBt = const Text('로그인');

  final controller = WebviewMainController("http://work.kotech.co.kr/");

  @override
  void initState() {
    super.initState();
    notificationService.initializeNotifications(onSelectNotification);
    locationService.checkAlarm();
  }

  String startTime = "";
  String endTime = "";
  void onSelectNotification(NotificationResponse? payload) async {
    // 현재 시간을 가져옴
    DateTime now = DateTime.now();

    // 클릭한 시간을 초까지 포맷팅하여 문자열로 변환
    setState(() {
      startTime = "${now.hour}:${now.minute}:${now.second}";
      endTime = "${now.hour + 8}:${now.minute}:${now.second}";
    });
  }

  // Stopwatch stopwatch = Stopwatch()..start(); // Stopwatch 시작
  // stopwatch.stop(); // Stopwatch 정지

  void _checkAlarm2() {
    if (locationService.distanceInMeters! <= locationService.targetDistance) {
      notificationService.showNotification("title : 도착 ", "boby: !!!!");
      _stopMeasuringDistance();
    }
  }

  Offset _offset = const Offset(0, 0);
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Location'),
      ),
      body: Stack(
        children: [
          WebViewWidget(
            controller: controller.controller,
          ),
          GestureDetector(
            onPanUpdate: (details) {
              setState(() {
                _offset += details.delta;
              });
            },
            child: Stack(
              children: [
                Positioned(
                  left: _offset.dx,
                  top: _offset.dy,
                  child: Container(
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.7),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: <Widget>[
                        Text(
                          '출근시간: $startTime',
                          style: const TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 6), // 시작 시간과 종료 시간 사이에 간격 추가
                        Text(
                          '퇴근시간: $endTime',
                          style: const TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(height: 8), // 퇴근 시간과 남은 거리 텍스트 사이에 간격 추가
                        if (locationService.distanceInMeters != null)
                          Padding(
                            padding: const EdgeInsets.all(1.0),
                            child: Container(
                              decoration: BoxDecoration(
                                borderRadius: BorderRadius.circular(10),
                                color: Colors.blue,
                              ),
                              padding: const EdgeInsets.all(1.0),
                              child: Text(
                                '남은 거리: ${locationService.distanceInMeters!.toStringAsFixed(1)} M',
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontSize: 20,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ),
                          ),
                        Row(
                          children: [
                            ElevatedButton(
                              onPressed: () {
                                setState(() {
                                  // 버튼을 클릭할 때마다 거리 측정 상태를 토글합니다.
                                  isDistanceMeasuring = !isDistanceMeasuring;
                                  if (isDistanceMeasuring) {
                                    // 거리 측정을 시작합니다.
                                    _startMeasuringDistance();
                                  } else {
                                    _stopMeasuringDistance();
                                  }
                                });
                              },
                              style: ButtonStyle(
                                backgroundColor:
                                    MaterialStateProperty.all<Color>(
                                        Colors.blue.shade50),
                                shape: MaterialStateProperty.all<
                                    RoundedRectangleBorder>(
                                  RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(10),
                                  ),
                                ),
                              ),
                              child: buttonText,
                            ),
                            ElevatedButton(
                              onPressed: () {
                                controller.controller.runJavaScript('''
                                document.getElementById('userId').value = 'ulta';
                                document.getElementById('userPwd').value = 'qkrdmfxk1!';
                                actionLogin();
                              ''');
                              },
                              child: loginBt,
                            ),
                            // 여기에 다른 위젯을 추가할 수 있습니다.
                          ],
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  void _startMeasuringDistance() {
    // 거리 측정 상태가 true이면 _getCurrentLocation 함수를 주기적으로 호출합니다.
    if (isDistanceMeasuring) {
      buttonText = const Text('종료');
      setState(() {
        locationService.getCurrentLocation();
        _checkAlarm2();
      });
      Future.delayed(const Duration(seconds: 1), _startMeasuringDistance);
    }
  }

  void _stopMeasuringDistance() {
    // 거리 측정 상태가 false이면 거리 측정을 중지합니다.
    buttonText = const Text('시작');
    isDistanceMeasuring = false;
  }
}
