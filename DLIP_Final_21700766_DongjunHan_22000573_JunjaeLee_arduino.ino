#include <FastLED.h>

#define LED_PIN_SCORE     27  // 점유율용
#define LED_PIN_EVENT     26  // 이벤트/세레모니용

#define NUM_LEDS          24
#define BRIGHTNESS        100
#define LED_TYPE          WS2812B
#define COLOR_ORDER       GRB

CRGB leds_score[NUM_LEDS];
CRGB leds_event[NUM_LEDS];

int red_percent = 50;
int blue_percent = 50;

String lastCommand = "";
unsigned long last_event_time = 0;

void setup() {
  FastLED.addLeds<LED_TYPE, LED_PIN_SCORE, COLOR_ORDER>(leds_score, NUM_LEDS);
  FastLED.addLeds<LED_TYPE, LED_PIN_EVENT, COLOR_ORDER>(leds_event, NUM_LEDS);
  FastLED.setBrightness(BRIGHTNESS);
  Serial.begin(115200);
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input == "GOAL" || input == "THROWIN" || input == "OUT") {
      lastCommand = input;
      last_event_time = millis();
    } else if (input.indexOf(',') > 0) {
      lastCommand = "";
      int sep = input.indexOf(',');
      red_percent = input.substring(0, sep).toInt();
      blue_percent = input.substring(sep + 1).toInt();
      displayPossession();
    }
  }

  handleEventLEDs();
}

// 점유율 LED 표시
void displayPossession() {
  int red_leds = round((red_percent / 100.0) * NUM_LEDS);
  for (int i = 0; i < NUM_LEDS; i++) {
    leds_score[i] = (i < red_leds) ? CRGB::Red : CRGB::Blue;
  }
  FastLED.show();  // 점유율 LED만 업데이트
}

// 이벤트 LED 처리
void handleEventLEDs() {

  if (lastCommand == "GOAL") {
    uint8_t t = millis() / 2;
    for (int i = 0; i < NUM_LEDS; i++) {
      leds_event[i] = CHSV((t + i * 10) % 255, 255, 255);
    }
  } 
else if (lastCommand == "THROWIN") {
  // 형광초록 고정
  fill_solid(leds_event, NUM_LEDS, CRGB(57, 255, 20));
} 
else if (lastCommand == "OUT") {
  // 형광보라 고정
  fill_solid(leds_event, NUM_LEDS, CRGB(255, 0, 255));
} 
else {
  // 이벤트 없음: 흰색
  fill_solid(leds_event, NUM_LEDS, CRGB::White);
}


  // // 이벤트 종료 후 자동 리셋 (5초 후 흰색)
  // if (lastCommand != "" && millis() - last_event_time > 5000) {
  //   lastCommand = "";
  //   fill_solid(leds_event, NUM_LEDS, CRGB::White);
  // }

  FastLED.show();  // 이벤트 LED만 업데이트
}
