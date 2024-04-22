import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'dart:math';

int successProbability = 5;
int maxTrials = 100;

class AnalysisWidget1 extends StatefulWidget {
  const AnalysisWidget1({super.key});

  @override
  AnalysisWidgets createState() => AnalysisWidgets();
}

class AnalysisWidgets extends State<AnalysisWidget1> {
  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        SizedBox(
          height: MediaQuery.of(context).size.height * 0.1,
          child: Row(
            children: [
              Expanded(
                flex: 1,
                child: Row(
                  children: [
                    SizedBox(
                      width: MediaQuery.of(context).size.width * 0.05,
                    ),
                    const Expanded(
                      child: Text(
                        '1회 시행 성공 확률(%) : ',
                        textAlign: TextAlign.right,
                        style: TextStyle(fontSize: 17),
                      ),
                    ),
                    Expanded(
                      child: TextField(
                        keyboardType: TextInputType.number,
                        onChanged: (value) {
                          setState(() {
                            successProbability = int.tryParse(value) ?? 5;
                          });
                        },
                        decoration: const InputDecoration(
                          hintText: '1회 시행 성공 확률',
                          hintStyle: TextStyle(fontSize: 12),
                          border: OutlineInputBorder(),
                          suffixText: '%',
                          suffixStyle: TextStyle(fontSize: 12),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              SizedBox(
                width: MediaQuery.of(context).size.width * 0.05,
              ),
              Expanded(
                flex: 1,
                child: Row(
                  children: [
                    const Text(
                      '시행 횟수 : ',
                      style: TextStyle(fontSize: 17),
                    ),
                    Expanded(
                      child: TextField(
                        keyboardType: TextInputType.number,
                        onChanged: (value) {
                          setState(() {
                            maxTrials = int.tryParse(value) ?? 100;
                          });
                        },
                        decoration: const InputDecoration(
                          hintText: '시행 횟수',
                          hintStyle: TextStyle(fontSize: 12),
                          border: OutlineInputBorder(),
                          suffixText: '번',
                          suffixStyle: TextStyle(fontSize: 12),
                        ),
                      ),
                    ),
                    SizedBox(
                      width: MediaQuery.of(context).size.width * 0.15,
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
        SizedBox(
          height: MediaQuery.of(context).size.height * 0.5,
          child: ProbabilityGraph(
            maxTrials: maxTrials,
            successProbability: successProbability,
          ),
        ),
      ],
    );
  }
}

class ProbabilityGraph extends StatelessWidget {
  final int maxTrials;
  final int successProbability;

  const ProbabilityGraph(
      {super.key, required this.maxTrials, required this.successProbability});

  @override
  Widget build(BuildContext context) {
    List<FlSpot> spots = [];

    int minTrials = 1;
    double cumulativeProbability = 0;

    for (int i = minTrials; i <= maxTrials; i++) {
      double probability = pow((1 - successProbability / 100), (i - 1)) *
          successProbability /
          100;
      cumulativeProbability += probability;
      FlSpot spot = FlSpot(i.toDouble(), cumulativeProbability);
      spots.add(spot);
    }
    return LineChart(
      LineChartData(
        gridData: const FlGridData(show: false),
        titlesData: const FlTitlesData(show: true),
        borderData: FlBorderData(show: true),
        minX: minTrials.toDouble(),
        maxX: maxTrials.toDouble(),
        minY: 0,
        maxY: 1,
        lineBarsData: [
          LineChartBarData(
            spots: spots,
            isCurved: true,
            dotData: const FlDotData(show: true),
            belowBarData: BarAreaData(show: false),
          ),
        ],
        lineTouchData: LineTouchData(
          touchTooltipData: LineTouchTooltipData(
            //tooltipBgColor: Colors.blue.withOpacity(0.8),
            getTooltipItems: (touchedSpots) {
              return touchedSpots.map((LineBarSpot touchedSpot) {
                final trials = touchedSpot.x.toInt();
                final probability = touchedSpot.y.toStringAsFixed(3);
                return LineTooltipItem(
                  '시행 횟수: $trials\n확률: $probability',
                  const TextStyle(color: Colors.white),
                );
              }).toList();
            },
          ),
        ),
      ),
    );
  }
}
