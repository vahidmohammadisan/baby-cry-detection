#include <Servo.h>

Servo myServo;
int servoPin = 9;

void setup() {
  Serial.begin(9600);
  myServo.attach(servoPin);
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    
    if (command == '1') {
      myServo.write(180);
      delay(1000);
      
      myServo.write(0);
    }
  }
}
