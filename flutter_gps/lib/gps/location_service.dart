// location_service.dart
import 'package:geolocator/geolocator.dart';

class LocationService {
  final double targetLatitude;
  final double targetLongitude;
  final double targetDistance;
  double? distanceInMeters;
  LocationService({
    required this.targetLatitude,
    required this.targetLongitude,
    required this.targetDistance,
  });
  Future<Position> determinePosition() async {
    try {
      Position position = await Geolocator.getCurrentPosition(
        desiredAccuracy: LocationAccuracy.high,
      );
      return position;
    } catch (e) {
      return Future.error(e);
    }
  }

  Future<Position> getCurrentLocation() async {
    try {
      Position position = await determinePosition();
      //checkAlarm(position.latitude, position.longitude);
      return position;
    } catch (e) {
      return Future.error(e);
      //print("Error: $e");
    }
  }

  Future<bool> checkAlarm() async {
    Future<Position> position = getCurrentLocation();
    bool isDistanceMeasuring = true;
    position.then((position) {
      distanceInMeters = Geolocator.distanceBetween(
        position.latitude,
        position.longitude,
        targetLatitude,
        targetLongitude,
      );

      if (distanceInMeters! <= targetDistance) {
        isDistanceMeasuring = !isDistanceMeasuring;
      }
    });
    return isDistanceMeasuring;
  }
}
