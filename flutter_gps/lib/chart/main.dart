import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'chart1.dart';
import 'chart2.dart';

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    SystemChrome.setPreferredOrientations([
      DeviceOrientation.landscapeLeft,
      DeviceOrientation.landscapeRight,
    ]);
    return const MaterialApp(
      title: 'Probability Distribution Graph',
      home: ProbabilityGraphScreen(),
    );
  }
}

class ProbabilityGraphScreen extends StatefulWidget {
  const ProbabilityGraphScreen({super.key});

  @override
  _ProbabilityGraphScreenState createState() => _ProbabilityGraphScreenState();
}

class _ProbabilityGraphScreenState extends State<ProbabilityGraphScreen> {
  String selectedAnalysis = '분석 1';
  Widget bodyWidget = Container(); // 기본적으로 빈 컨테이너를 사용

  @override
  Widget build(BuildContext context) {
    Widget bodyWidget = Container();
    // 선택된 분석에 따라 bodyWidget을 업데이트
    if (selectedAnalysis == '분석 1') {
      bodyWidget = const AnalysisWidget1();
    } else if (selectedAnalysis == '분석 2') {
      bodyWidget = const LinearRegressionScreen();
    } else if (selectedAnalysis == '분석 3') {
      // bodyWidget = Analysis3Widget();
    }
    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: [
            PopupMenuButton<String>(
              initialValue: selectedAnalysis,
              itemBuilder: (BuildContext context) => <PopupMenuEntry<String>>[
                const PopupMenuItem<String>(
                  value: '분석 1',
                  child: Text('분석 1'),
                ),
                const PopupMenuItem<String>(
                  value: '분석 2',
                  child: Text('분석 2'),
                ),
                const PopupMenuItem<String>(
                  value: '분석 3',
                  child: Text('분석 3'),
                ),
                // 원하는 만큼 메뉴 항목 추가
              ],
              onSelected: (String value) {
                setState(() {
                  selectedAnalysis = value; // 선택된 항목 업데이트
                });
              },
            ),
            const SizedBox(width: 8), // 선택한 항목과 메뉴 사이 간격
            Text(selectedAnalysis), // 선택한 항목 표시
          ],
        ),
      ),
      body: bodyWidget,
    );
  }
}
