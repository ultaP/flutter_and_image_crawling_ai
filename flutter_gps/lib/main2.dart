import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Probability Distribution Graph',
      home: Scaffold(
        appBar: AppBar(
          title: Text('Probability Distribution Graph'),
        ),
        body: Center(
          child: ProbabilityGraph(),
        ),
      ),
    );
  }
}

class ProbabilityGraph extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return LineChart(
      LineChartData(
        gridData: FlGridData(show: false),
        titlesData: FlTitlesData(show: false),
        borderData: FlBorderData(show: true),
        minX: 1,
        maxX: 10,
        minY: 0,
        maxY: 0.1,
        lineBarsData: [
          LineChartBarData(
            spots: [
              FlSpot(1, 0.01),
              FlSpot(2, 0.01),
              FlSpot(3, 0.01),
              // 나머지 시행횟수에 대한 확률 추가
            ],
            isCurved: true,
            //color: Color,
            dotData: FlDotData(show: true),
            belowBarData: BarAreaData(show: false),
          ),
        ],
      ),
    );
  }
}
