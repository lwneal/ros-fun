@0xa89ff8a8e218a3ba;

enum VisionRequestType {
    resNet50 @0;
    detectHuman @1;
}

struct RobotCommand {
    headRelAzumith @0 :Float64;
    headRelAltitude @1 :Float64;
    descriptiveStatement @2 :Text;
}

struct FrameMsg {
    width @0 :Int32;
    height @1 :Int32;
    timestampEpoch @2 :Float64;
    frameData @3 :Data;

    visionType @4 :VisionRequestType;
    robotCommand @5 :RobotCommand;
}
