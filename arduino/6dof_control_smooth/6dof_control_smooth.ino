// #include <Servo.h>
// #include <Wire.h>
#include <Adafruit_PWMServoDriver.h>


Adafruit_PWMServoDriver pca9685 = Adafruit_PWMServoDriver(0x40);
#define SERVOMIN 105
#define SERVOMAX 510

#define us_per_sec 1000000

class Servo2 {
  public:
    int angle;
    int min_angle, max_angle;
    int min_us, max_us;
    int pin;
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
      angle = angle;
      pca9685.setPWM(pin, 0, us);
    }
    int read(){
      return map(pca9685.getPWM(pin, true), min_us, max_us, min_angle, max_angle);
    }
};


/*
MG996r max speed:
0.17s/60deg = 352 deg/sec
DS3245sg max speed:
0.18sec/60deg = 333 deg/sec
*/

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

int servo_parameters[3][4] = {
  {0, 180, SERVOMIN, SERVOMAX},
  {0, 270, SERVOMIN, 530},
  {0, 180, 100, 510}
};

int servo_types[7] = {0, 1, 1, 0, 2, 1, 1};

int gear_offsets[7] = {98,97,86,90,83,138,0};
// int mechanical_offsets[7] = {0, -4, -20, 0, 0, 0, 0};
int mechanical_offsets[7] = {0, 0, 0, 0, 0, 0, 0};
int rotation[7] = {1,1,1,1,-1,1,1};

const int VECTOR_SIZE = 7;  // Define the size of the action vector
double action_vector[VECTOR_SIZE];  // Store parsed values
int desired_angles[7] = {0,0,0,0,0,0,0};

unsigned long time_now = 0;
unsigned long time_diff = 0;

// int desired_angles[7] = {90, 97,86,90,93,90,1};

int max_degree_per_sec = 90;
const int interval_num = 50;
const int period = 10000;
int initial_angles[7];
int cur_interval = 0;
int max_interval = 0;
int delta_angles[7];

void setup() {
  pca9685.begin();
  pca9685.setPWMFreq(50);
  Serial.begin(115200);
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
    unsigned long test1 = micros();
    String receivedString = Serial.readStringUntil('\n');  // Read the incoming data until newline - takes ~ 3ms
    parseActionVector(receivedString);    // takes ~ 3ms
    for(int i = 0; i < servos_len-1; i++){
      desired_angles[i] = (rotation[i] * rad2deg(action_vector[i])) + gear_offsets[i] + mechanical_offsets[i];
    }
    int gripper_angle = (int)map(action_vector[6]*1000, 200, 800, gripper_min, gripper_max);
    gripper_angle = constrain(gripper_angle, gripper_min, gripper_max);
    desired_angles[servos_len-1] = gripper_angle;
    
    long max_duration = 0;
    long duration = 0;
    for(int i = 0; i < servos_len; i++){
      initial_angles[i] = servos[i].read();
      delta_angles[i] = desired_angles[i] - initial_angles[i];
      duration = abs(delta_angles[i]) * 1L * us_per_sec / max_degree_per_sec * 1L;
      max_duration = max(max_duration, duration);
    }
    unsigned long test2 = micros() - test1;
    max_interval = (int) (max_duration / period + 1);

    cur_interval = 1;
    
  }
  time_diff = micros() - time_now;
  if(time_diff >= period && cur_interval < max_interval){
    for(int i = 0; i < servos_len; i++){
      servos[i].write(initial_angles[i] + ((delta_angles[i] * cur_interval) / max_interval));
    }
    cur_interval++;
    if(cur_interval >= max_interval){
      Serial.println("Action Done");
    }
    time_now = micros();
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