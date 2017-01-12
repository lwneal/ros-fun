@0xa89ff8a8e218a3ba;

enum VisionRequestType {
    resNet50 @0;
    detectHuman @1;
}

struct FrameMsg {
    width @0 :Int32;
    height @1 :Int32;
    timestampEpoch @2 :Float64;
    frameData @3 :Data;

    visionType @4 :VisionRequestType;
}
