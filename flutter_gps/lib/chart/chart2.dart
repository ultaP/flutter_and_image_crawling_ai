import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'dart:math';

class LinearRegressionScreen extends StatefulWidget {
  const LinearRegressionScreen({super.key});

  @override
  _LinearRegressionScreenState createState() => _LinearRegressionScreenState();
}

class _LinearRegressionScreenState extends State<LinearRegressionScreen> {
  List<double> xValues = [1, 2, 3, 4, 5];
  List<double> yValues = [2, 3, 4, 5, 6];

  double slope = 0;
  double intercept = 0;

  @override
  void initState() {
    super.initState();
    calculateLinearRegression();
  }

  void calculateLinearRegression() {
    double sumX = xValues.reduce((value, element) => value + element);
    double sumY = yValues.reduce((value, element) => value + element);
    double sumXY = 0;
    double sumXSquare = 0;

    for (int i = 0; i < xValues.length; i++) {
      sumXY += xValues[i] * yValues[i];
      sumXSquare += xValues[i] * xValues[i];
    }

    slope = (xValues.length * sumXY - sumX * sumY) /
        (xValues.length * sumXSquare - sumX * sumX);
    intercept = (sumY - slope * sumX) / xValues.length;

    print('Slope: $slope');
    print('Intercept: $intercept');
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        AspectRatio(
          aspectRatio: 1.7,
          child: LineChart(
            LineChartData(
              titlesData: const FlTitlesData(),
              borderData: FlBorderData(show: true),
              minX: xValues.first,
              maxX: xValues.last,
              minY: yValues.reduce((value, element) => min(value, element)),
              maxY: yValues.reduce((value, element) => max(value, element)),
              lineBarsData: [
                LineChartBarData(
                  spots: List.generate(
                    xValues.length,
                    (index) => FlSpot(xValues[index], yValues[index]),
                  ),
                  isCurved: true,
                  //colors: [Colors.blue],
                  dotData: const FlDotData(show: false),
                ),
                LineChartBarData(
                  spots: [
                    FlSpot(xValues.first, slope * xValues.first + intercept),
                    FlSpot(xValues.last, slope * xValues.last + intercept),
                  ],
                  isCurved: true,
                  //colors: [Colors.red],
                  dotData: const FlDotData(show: false),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}
