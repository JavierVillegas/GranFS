#pragma once

#include "ofMain.h"
#include "ofxMidi.h"
#include "ofxOpenCv.h"



class testApp : public ofBaseApp, public ofxMidiListener {
public:
    void setup();
    void update();
    void draw();
    void exit();
    
    void keyPressed(int key);
    void keyReleased(int key);
    void mouseMoved(int x, int y);
    void mouseDragged(int x, int y, int button);
    void mousePressed(int x, int y, int button);
    void mouseReleased(int x, int y, int button);
    void windowResized(int w, int h);
    void dragEvent(ofDragInfo dragInfo);
    void gotMessage(ofMessage msg);
    
    void FillGrainList(ofVec3f); // fill the grain list
    void FillGrainList(float); // fill the grain list
  
    ofVideoGrabber 		vidGrabber1;
    ofVideoGrabber 		vidGrabber2;
    ofxCvColorImage			colorImg;
    ofxCvColorImage         colorImg2;
    ofxCvGrayscaleImage 	grayImage;
    
    void newMidiMessage(ofxMidiMessage& eventArgs);
    void audioIn(float *Ainput, int BufferSize, int nChannels);
    
	
	stringstream text;
	
	ofxMidiIn	midiIn;
	ofxMidiMessage midiMessage;
    //Audio variables:
    
   // vector<float> AudioValues;
    float CurrentEnergy;
    ofSoundStream soundStream;
    
    // Face
    
    void detectFace( cv::Mat frame );
    void detectMotion (int Nprev);
    void EnergyCenterSelect(void);
    void AudioRandomSelect(void);
    
};
