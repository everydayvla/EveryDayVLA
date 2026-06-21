// #include <Servo.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>


Adafruit_PWMServoDriver pca9685 = Adafruit_PWMServoDriver(0x40);
#define SERVOMIN 105
#define SERVOMAX 510


class Servo2 {
  public:
    int angle;
    int min_angle, max_angle;
    int min_us, max_us;
    int pin;
    // Servo servo;
    Servo2(){
      pin = -1;
      min_angle = 0;
      max_angle = 180;
      min_us = 1000;
      max_us = 2000;
    }
    Servo2(int pin, int min_angle, int max_angle, int min_us, int max_us): pin(pin), min_angle(min_angle), max_angle(max_angle), min_us(min_us), max_us(max_us)
    {
    }
    void write(int angle){
      int us =  (int) map(angle, min_angle, max_angle, min_us, max_us);

      pca9685.setPWM(pin, 0, us);
    }
    int read(){
      return map(pca9685.getPWM(pin, true), min_us, max_us, min_angle, max_angle);
    }
};

int servos_len = 7;
# define rad2deg(rad) (int) (rad*180/PI)
const int gripper_min = 10;     // closed grip
const int gripper_max = 110;    // open grip

Servo2 servos[7];
int servo_pins[7] = {0,1,2,3,4,5,6};
# define base servos[0]
# define shoulder servos[1]
# define elbow servos[2]
# define wrist1 servos[3]
# define wrist2 servos[4]
# define wrist3 servos[5]
# define gripper servos[6]

int servo_parameters[2][4] = {
  {0, 180, SERVOMIN, SERVOMAX},
  {0, 270, SERVOMIN, 530}
};

int servo_types[7] = {0, 1, 1, 0, 0, 1, 1};

int gear_offsets[7] = {98,97,86,90,93,138,0};
int mechanical_offsets[7] = {0, 0, 0, 0, 0, 0, 0};
int rotation[7] = {1,1,1,1,-1,1,1};

const int VECTOR_SIZE = 7;  // Define the size of the action vector
double action_vector[VECTOR_SIZE];  // Store parsed values
int angles_global[7] = {0,0,0,0,0,0,0};

void setup() {
  pca9685.begin();
  pca9685.setPWMFreq(50);
  Serial.begin(9600);
  Serial.println("Start");
  for(int i = 0; i < servos_len; i++){
    servos[i] = Servo2(servo_pins[i], servo_parameters[servo_types[i]][0], servo_parameters[servo_types[i]][1], servo_parameters[servo_types[i]][2], servo_parameters[servo_types[i]][3]);
  }
  base.write(90);
  shoulder.write(90);
  elbow.write(90);
  wrist1.write(90);
  wrist2.write(90);
  wrist3.write(135);
  gripper.write(gripper_max);
  delay(1000);
  Serial.println("Done with setup() function!");
}

void loop() {
  if (Serial.available()) {
      String receivedString = Serial.readStringUntil('\n');  // Read the incoming data until newline
      parseActionVector(receivedString);

      for(int i = 0; i < servos_len-1; i++){
        angles_global[i] = (rotation[i] * rad2deg(action_vector[i])) + gear_offsets[i] + mechanical_offsets[i];
      }
      int gripper_angle = (int)map(action_vector[6]*1000, 200, 800, gripper_min, gripper_max);
      gripper_angle = constrain(gripper_angle, gripper_min, gripper_max);
      
      angles_global[servos_len-1] = gripper_angle;

      run_over_time(angles_global, 200);
      Serial.println("Action Done");
    }
}

void parseActionVector(String input) {
    int index = 0;
    char *ptr = strtok((char*)input.c_str(), ",");  // Tokenize the string using commas
    while (ptr != NULL && index < VECTOR_SIZE) {
        action_vector[index] = atof(ptr);  // Convert string to float
        ptr = strtok(NULL, ",");  // Get the next value
        index++;
    }
}

void run_over_time(int* angles, int time){
  int cur_angles[servos_len];
  for(int i = 0; i < servos_len; i++){
    cur_angles[i] = servos[i].read();
  }
  for(int i = 0; i < time; i++){
    for(int j = 0; j < servos_len; j++){
      int angle = map(i, 0, time, cur_angles[j], angles[j]);
      servos[j].write((int)angle);
    }
  }
}