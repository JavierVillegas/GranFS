#include "testApp.h"


const int Nx = 352;
const int Ny = 288;
// A vector with all the frames.
vector <cv::Mat> TheFramesInput;

vector<ofVec3f> FirstGrains;
vector<ofVec3f> SecondGrains;
vector<ofVec3f> OutGrains;
vector<ofVec3f> GrainMask;
vector<float> randAng;


const int Nlast = 400;
ofVec3f BlockDims;
float inOp;//  overlaping porcentage
float outOp;//  overlaping porcentage

int Gs = 51; // for fixed grain size in all dimensions
cv::Mat CurrentOutput;
int fc=0; // frame counter


void InitGrains(void);

float CalculateScalecorrect(void);
float ScaleCorrectFactor=1.0;

int circuIndex = Nlast-1;


// Control global variables:
float SLD1 = 0;
float SLD2 = 0;
float SLD3 = 0;
float SLD4 = 0;
float SLD5 = 1.0;
float SLD6 = 0.0;
float SLD7 = 0.0;
float SLD8 = 32.0;
float KNOBMOTION = 60;

// booleans for all the effects:

// Rotation
bool boolRotGrains = false;
// Time delay
bool boolDelaygrains = false;
// non linear geometric
bool boolGeomGrains = false;
// space shifting
bool boolShiftGrain = false;
// camera 2
bool boolCam2Grain = false;

bool boolhueshift = false;
bool boolSatChange = false;

// modes to select grains
int SelectionMode =5; // All mutual exclusive
                    // 0 no grains
                    // 1 all grains
                    // 2 Random 50/50
                    // 3 Motion detection
                    // 4 Frontal face detection
                    // 5 Audio circle
                    // 6 Audio random


// Control Mode variables

int RotMode = 0; // 0 fixed
                 // 1 random
                 // 2 Audio 

int DelayMode = 0; // 0 fixed
                   // 1 random
                   // 2 Audio 
int GeomMode = 0;  // 0 fixed
                   // 1 random
                   // 2 Audio 
int ShiftMode = 0;  // 0 fixed
                    // 1 random
                    // 2 Audio 
int HueMode = 0;  // 0 fixed
                    // 1 random
                    // 2 Audio 



// camera 2 has no control modes







cv::Mat Cam2Frame;

// Audio

int BuffS = 512;

float EnergyBuffer[4] = {0.0,0.0,0.0,0.0};
int EnergyIndy = 0;


// Face detect variables

cv::CascadeClassifier face_cascade;
cv::Rect FirstFace;


vector<int> GrainsRandom;

// extra bools

bool randomReady = false;


int TheAudioDevice = 2;
// 0 Buid in mic

//2 locitech cam mic 


int ModeSelector = 1;





//--------------------------------------------------------------
void testApp::setup(){
   
    // First creating an empty array
    
    BlockDims.x = Nx;
    BlockDims.y = Ny;
    BlockDims.z = Nlast;
    cv::Mat AuxMat(Ny,Nx,CV_8UC3,cv::Scalar(0,0,0));
    for (int k = 0; k< Nlast; k++){
        TheFramesInput.push_back(AuxMat.clone());
    }
    
    
   // vidGrabber.setVerbose(true);
 //   vidGrabber.initGrabber(Nx,Ny);
    
    ofSetFrameRate(30);
    ofHideCursor();
    inOp =50;
    outOp =50;
   

    InitGrains();
    

    
    
    
//    UpdateSchedulerUpDown(TheUf,1);
    ofSetLogLevel(OF_LOG_VERBOSE);
    vidGrabber1.listDevices();
    vidGrabber1.setDeviceID(1);
    vidGrabber1.initGrabber(Nx,Ny);
    vidGrabber2.setDeviceID(0);
    vidGrabber2.initGrabber(Nx,Ny);

    //vidGrabber.initGrabber(Nx,Ny);

//    std::exit(1);
    colorImg.allocate(Nx,Ny);
	grayImage.allocate(Nx,Ny);
    
    
    // Midi
    
    // open port by number
	midiIn.openPort(0);
	//midiIn.openPort("IAC Pure Data In");	// by name
	//midiIn.openVirtualPort("ofxMidiIn Input");	// open a virtual port
	
	// don't ignore sysex, timing, & active sense messages,
	// these are ignored by default
	midiIn.ignoreTypes(false, false, false);
	
	// add testApp as a listener
	midiIn.addListener(this);
	
	// print received messages to the console
	midiIn.setVerbose(false);
    
    
    
    // scale correction  factor 50% overlap
    ScaleCorrectFactor =CalculateScalecorrect();
    
    
    //AudioValues.assign(BuffS, 0.0);
    
    
    cout<<"aca van: "<<endl;
    soundStream.listDevices();
    
    // selecting the microhpone
   // soundStream.setup(this, 0, 2, 44100, BuffS, 4);
    
    //selecting the logitech camera mic device 2
    soundStream.setDeviceID(TheAudioDevice);
 
    if (TheAudioDevice ==0){
     soundStream.setup(this, 0, 2, 44100, BuffS, 4);
    }
    else if(TheAudioDevice==2){
    soundStream.setup(this, 0, 1, 48000, BuffS, 4);
    }
    
    // load cascade detector files
    
    //-- 1. Load the cascades
    if( !face_cascade.load( "/Users/jvillegas/of_v0.7.4_osx_release/apps/myApps/GrainModesFS/haarcascade_frontalface_alt_tree.xml" ) ){
        // if( !face_cascade.load( "/Users/javiervillegas/of_v0.7.4_osx_release/apps/myApps/FaceCenter/haarcascade_mcs_eyepair_big.xml" ) ){
        //  if( !face_cascade.load( "/Users/javiervillegas/of_v0.7.4_osx_release/apps/myApps/FaceCenter/haarcascade_mcs_nose.xml" ) ){
        
        cout<<"--(!)Error loading cara \n"<<endl;
    };
    
    
    int Osamp = (int)(ceil(Gs*inOp/100.0));
    int jp = Gs - Osamp;
    
    
    
    // fixed output size
    int outOsamp = (int)(ceil(Gs*outOp/100.0));
    int ojp = Gs - outOsamp;
    int outYsize = Gs + ojp*((BlockDims.y-Gs)/jp);
    int outXsize = Gs + ojp*((BlockDims.x-Gs)/jp);
    
    // creating an empty frame for the output
    
    CurrentOutput =cv::Mat::zeros(outYsize,outXsize, CV_8UC3);
    
    
    
}

//--------------------------------------------------------------

void testApp::update(){
    
    // setups for each piece
    switch (ModeSelector) {
            
        case 1:
            SelectionMode =6;
            // rotation with random selection
            boolGeomGrains = false;
            boolhueshift = false;
            boolSatChange = false;
            
            boolRotGrains = true;
            boolDelaygrains = false;
            boolShiftGrain = false;
            boolCam2Grain = false;
            boolSatChange = false;
            
            
            RotMode = 2;
            
            SLD1 = 92;
            SLD7 = 6;
        
            
        break;
            
            
            
        case 2:
            SelectionMode =1;
        
            boolGeomGrains = false;
            boolhueshift = true;
            boolSatChange = false;
            
            boolRotGrains = false;
            boolDelaygrains = true;
            boolShiftGrain = false;
            boolCam2Grain = false;
            boolSatChange = true;
            
            
            DelayMode = 1;
            HueMode = 1;

            SLD2 = 6;
            SLD3 = 73;
            SLD4 = 59;
            
      
            break;
            
        case 3:
            SelectionMode =2;// 3 if face doesnt work 2 if motion doesnt
         
            boolGeomGrains = false;
            boolhueshift = false;
            boolSatChange = false;
            
            boolRotGrains = false;
            boolDelaygrains = false;
            boolShiftGrain = true;
            boolCam2Grain = false;
            boolSatChange = true;
            
            ShiftMode = 2;
             
            SLD4 = 80;
            SLD6 = 42;
                        
  
            break;            

            
        case 4:
            SelectionMode =5;
 
            boolGeomGrains = false;
            boolhueshift = false;
            boolSatChange = false;
            
            boolRotGrains = false;
            boolDelaygrains = false;
            boolShiftGrain = false;
            boolCam2Grain = true;
            boolSatChange = false;

            
            SLD7 = 84;
        
            
      
            break;             
            

        case 5:
            SelectionMode =5;
            
            boolGeomGrains = false;
            boolhueshift = false;
            boolSatChange = false;
            
            boolRotGrains = true;
            boolDelaygrains = false;
            boolShiftGrain = false;
            boolCam2Grain = true;
            boolSatChange = false;
            
            RotMode = 0;
            
            SLD7 = 84;
            SLD1 = 64;
            
  
            break;              
            
        case 6:
            SelectionMode =2;
       
            boolGeomGrains = true;
            boolhueshift = false;
            boolSatChange = false;
            
            boolRotGrains = false;
            boolDelaygrains = false;
            boolShiftGrain = false;
            boolCam2Grain = false;
            boolSatChange = false;
            
            GeomMode =2;
            
            SLD5 = 70;
            
     
            break;          
  
        case 7:
            SelectionMode =3;
            
            boolGeomGrains = false;
            boolhueshift = false;
            boolSatChange = false;
            
            boolRotGrains = true;
            boolDelaygrains = false;
            boolShiftGrain = true;
            boolCam2Grain = false;
            boolSatChange = false;
            
            RotMode = 2;
            ShiftMode = 0;
            
                    
            SLD6 = 11;
            
            
            break;                
  
        case 8:
            SelectionMode =6;
            
            boolGeomGrains = false;
            boolhueshift = false;
            boolSatChange = false;
            
            boolRotGrains = false;
            boolDelaygrains = false;
            boolShiftGrain = false;
            boolCam2Grain = true;
            boolSatChange = false;
            
            SLD7 = 6;
            
            break; 
        case 9:
            SelectionMode =1;
            
            boolGeomGrains = true;
            boolhueshift = false;
            boolSatChange = false;
            
            boolRotGrains = false;
            boolDelaygrains = true;
            boolShiftGrain = false;
            boolCam2Grain = false;
            boolSatChange = false;
            
            GeomMode = 2;
            DelayMode = 1;
            
            
            SLD5 = 125;
            SLD2 = 12;
            
            break;   
            
        case 10:
            SelectionMode =3;
            
            boolGeomGrains = false;
            boolhueshift = false;
            boolSatChange = false;
            
            boolRotGrains = false;
            boolDelaygrains = false;
            boolShiftGrain = false;
            boolCam2Grain = true;
            boolSatChange = false;
            
            
            
            break;  


            
        case 11: // piece 11:
            // circular selection
            SelectionMode =5;
            // nonlinear and Hue
            boolGeomGrains = true;
            boolhueshift = true;
            boolSatChange = true;
            
            boolRotGrains = false;
            boolDelaygrains = false;
            boolShiftGrain = false;
            boolCam2Grain = false;
            boolSatChange = false;
            
            // Hue random  Geometrical fixed
            HueMode = 1;  
            GeomMode = 0;
            
            // Slider Values:

            SLD5 = 5;
            SLD7 = 47;
            SLD3 = 65;
            SLD4 = 59;
            
            break;
            
        case 12:
            SelectionMode =6;
            
            boolGeomGrains = true;
            boolhueshift = false;
            boolSatChange = false;
            
            boolRotGrains = false;
            boolDelaygrains = false;
            boolShiftGrain = false;
            boolCam2Grain = false;
            boolSatChange = false;
            
     
            GeomMode = 0;
            
            SLD5 = 39;
            SLD7 = 13;
            
            break;            

            
            
            
        default:
            break;
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
    bool bNewFrame = false;
    

    vidGrabber1.update();

        
    bNewFrame = vidGrabber1.isFrameNew();
   
    //
    if (bNewFrame){
        //
        // circular buffer

        colorImg.setFromPixels(vidGrabber1.getPixels(), Nx,Ny);
          
        cv::Mat AuxMat;
        AuxMat = colorImg.getCvImage();
        AuxMat.copyTo(TheFramesInput[circuIndex]);
        
        
        fc = circuIndex;
        // Fc has the current sample index
        circuIndex--;
        if (circuIndex<0){
            circuIndex = Nlast-1;
        }
        
        
        // Selection of grains:
        switch (SelectionMode) {
        
            case 0://none
                for (int g=0; g < GrainMask.size(); g++) {
                    GrainMask[g].z = 0.0;
                }
            break;
            
            case 1:// all
                for (int g=0; g < GrainMask.size(); g++) {
                    GrainMask[g].z = 1.0;
                }
            break;
            
            case 2: // random
            if (randomReady==false) {
                //cout<< "PASOOOO"<<ModeSelector<<endl;
                for (int g=0; g < GrainMask.size(); g++) {
                    GrainMask[g].z = (ofRandom(1.0)> 0.5);
                    }
              randomReady = true;       
            }
            break;              
            
            case 3:   // Motion Detection  
                detectMotion(2);
            break;
                
            case 4:
                 detectFace(AuxMat);
                 for (int g=0; g < GrainMask.size(); g++) {
                     if((GrainMask[g].x+Gs/2.0>FirstFace.x)&&(GrainMask[g].y+Gs/2.0>FirstFace.y)&&
                        (GrainMask[g].x+Gs/2.0<(FirstFace.x + FirstFace.width))&&
                        (GrainMask[g].y+Gs/2.0<(FirstFace.y + FirstFace.height)))
                        {
                          GrainMask[g].z = 1.0;
                        }
                   else{
                          GrainMask[g].z = 0.0;
                        }
                  }
            break;
        
            case 5:
               EnergyCenterSelect();
            break;
        
            case 6:
                AudioRandomSelect();
            break;
        }
        
        // Getting a frame from the other camera
        vidGrabber2.update();
       // if (vidGrabber2.isFrameNew()){
        colorImg2.setFromPixels(vidGrabber2.getPixels(),Nx,Ny);
        Cam2Frame = colorImg2.getCvImage();
     //   }
        
            
        
    // calculate new output size:
    int outOsamp = (int)(ceil(Gs*outOp/100.0));
    int ojp = Gs - outOsamp;
 
    // input jump size
    int Osamp = (int)(ceil(Gs*inOp/100.0));
    int jp = Gs - Osamp;

    
    int outYsize = Gs + ojp*((BlockDims.y-Gs)/jp);
    int outXsize = Gs + ojp*((BlockDims.x-Gs)/jp);
    
   // creating an empty frame for the output

    CurrentOutput =cv::Mat::zeros(outYsize,outXsize, CV_8UC3);
        
        
        
        // Delay effect:
        
        if (boolDelaygrains){
            
            for (int g =0; g < SecondGrains.size(); g++) {
                if(GrainMask[g].z ==1.0){
                    if (DelayMode == 0){
                        SecondGrains[g].z = (int)((SLD2/127.0)*(Nlast/2.0));
                        
                    }
                    else if(DelayMode == 1){
                        SecondGrains[g].z = (int)(ofRandom((SLD2/127.0)*(Nlast/2.0)));
                    }
                    else if(DelayMode == 2){
                        SecondGrains[g].z = (int)(sqrtf(CurrentEnergy)*(SLD2/127.0)*(Nlast/2.0));
                    }
                }
                else{
                    SecondGrains[g].z = 0.0;
                }
            }
            
            
        }  
        
        else{
            for (int g =0; g < SecondGrains.size(); g++) {

            SecondGrains[g].z = 0.0;
            }
        }
 
    // pointer to output data:
    
    unsigned char *output = (unsigned char*)(CurrentOutput.data);
    
    // running throught the list of grains
    
    for (int g =0; g < FirstGrains.size(); g++) {
        
        
        // two halves
        for (int m=0; m<2; m++) {
          // copy the grain from the input
            int IntiFrame;
            if(m==1){
                    IntiFrame = (fc + (int)FirstGrains[g].z)%Nlast;
             }
            else{
                    IntiFrame = ((fc + (int)SecondGrains[g].z)%Nlast);
            }
            
            //int IntiFrame = (m==1)?(fc + (int)FirstGrains[g].z)%Nlast:((fc + (int)SecondGrains[g].z)%Nlast);
            
            unsigned char *input = (unsigned char*)(TheFramesInput[IntiFrame].data);
            unsigned char *input2 = (unsigned char*)(Cam2Frame.data);
            
            // Hue and saturation transformations:
            
            
            float Ang;
            if (HueMode ==0) {Ang = SLD3/127.0*2*PI;}
            if (HueMode ==1) {Ang = ofRandom(SLD3/127.0*2*PI);}
            if (HueMode ==2) {Ang = sqrtf(CurrentEnergy)*SLD3/127.0*2*PI;}
            
            // Rotation effect
            
            float RotFact;
            if (RotMode ==0){RotFact =PI*SLD1/127.0;}
            else if (RotMode ==1){RotFact =ofRandom(PI*SLD1/127.0);}
            else if (RotMode ==2){RotFact =sqrtf(CurrentEnergy)*PI*SLD1/127.0;}
            
         // Geometric distortion
            float GeoAlpha;
            
            if (GeomMode == 0){GeoAlpha = 0.6+2*SLD5/127.0;}
            else if (GeomMode == 1){GeoAlpha = 0.6 + ofRandom(2*SLD5/127.0);}
            else if (GeomMode == 2){GeoAlpha = (0.6+2*sqrtf(CurrentEnergy)*SLD5/127.0);}     
            
            
            // grain shift
            
            float shiftAmount;
            
            if (ShiftMode == 0) { shiftAmount = 5.0*SLD6/127.0;}
            if (ShiftMode == 1) { shiftAmount = ofRandom(5.0*SLD6/127.0);}
            if (ShiftMode == 2) { shiftAmount = sqrtf(CurrentEnergy)*5.0*SLD6/127.0;}  
            
            
            
            
            for (int x =0; x<Gs; x++) {
                for (int y=0; y<Gs; y++) {
                    
                    float rIn,gIn,bIn;
                    float rOut,gOut,bOut;
                    float Scale;
                    Scale = ((0.5*0.5*0.5)*(1.0 -cosf(2*PI*x/(float)(Gs-1)))*
                             (1.0 -cosf(2*PI*y/(float)(Gs-1)))*
                             (1.0 -cosf(2*PI*(m*(Gs-1)/2.0 +(fc%jp))/(float)(Gs-1))));
                    // m =0 is the next grain
                    
                    Scale/=(ScaleCorrectFactor*ScaleCorrectFactor*ScaleCorrectFactor);
                    // pixels at the inut
                    int yindIn;
                    int xindIn;
                    

                    
                    
                    
                    if ((!boolRotGrains)||(GrainMask[g].z==0.0)) {
                    
                        if (m==1) {
                            xindIn = x + FirstGrains[g].x;
                            yindIn = y + FirstGrains[g].y;
                        }
                        else {
                            xindIn = x + SecondGrains[g].x;
                            yindIn = y + SecondGrains[g].y;
                        }
                    }
                    if ((boolRotGrains)&&(GrainMask[g].z==1.0)){
                    
                        if (m==1) {
                            float newX = (x + FirstGrains[g].x) - (FirstGrains[g].x+Gs/2.0);
                            float newY = (y + FirstGrains[g].y) - (FirstGrains[g].y+Gs/2.0);
                            xindIn = (int)(newX*cos(RotFact) - newY*sin(RotFact)) + (FirstGrains[g].x+Gs/2.0);
                            yindIn = (int)(newX*sin(RotFact) + newY*cos(RotFact)) + (FirstGrains[g].y+Gs/2.0);
                        }
                        else {
                            float newX = (x + SecondGrains[g].x) - (SecondGrains[g].x+Gs/2.0);
                            float newY = (y + SecondGrains[g].y) - (SecondGrains[g].y+Gs/2.0);
                            xindIn = (int)(newX*cos(RotFact) - newY*sin(RotFact)) + (SecondGrains[g].x+Gs/2.0);
                            yindIn = (int)(newX*sin(RotFact) + newY*cos(RotFact)) + (SecondGrains[g].y+Gs/2.0);

                        }
                    
                    
                    }
                    
                    
                    // Geometrical transformations:
                    // center
                    
                    if ((boolGeomGrains)&&(GrainMask[g].z==1.0)){
                        
                        float newX,newY;
                        if (m==1) {
                            newX = (xindIn - FirstGrains[g].x - Gs/2.0);
                            newY = (yindIn - FirstGrains[g].y - Gs/2.0);
                        }
                        
                        else{
                            newX = (xindIn - SecondGrains[g].x - Gs/2.0);
                            newY = (yindIn - SecondGrains[g].y - Gs/2.0);
                        }

                        // First converting to polar and normalizing magnitude
                        
                        

                        
                        float Rf = sqrtf(2*(newX*newX+newY*newY))/Gs;
                        float newR = powf(Rf,GeoAlpha);
                    
                        newY = (Rf!=0)?(newR*newY/Rf):0.0;
                        newX = (Rf!=0)?(newR*newX/Rf):0.0;
                    
                        if (m==1) {
                            xindIn = (int)(newX) + (FirstGrains[g].x+Gs/2.0);
                            yindIn = (int)(newY) + (FirstGrains[g].y+Gs/2.0);
                        }
                    
                        else{
                            xindIn = (int)(newX) + (SecondGrains[g].x+Gs/2.0);
                            yindIn = (int)(newY) + (SecondGrains[g].y+Gs/2.0);
                        }
                    
                    }
                    
  
                    
                    xindIn = (xindIn<0)?0:xindIn;
                    xindIn = (xindIn>outXsize-1)?BlockDims.x:xindIn;
                    
                    yindIn = (yindIn<0)?0:yindIn;
                    yindIn = (yindIn>outYsize-1)?BlockDims.y:yindIn;
                    
                    
                    
                    
                    
                    if((boolCam2Grain)&&(GrainMask[g].z==1.0)){
                        bIn = (float)input2[(int)(3*BlockDims.x * (yindIn) + 3*(xindIn)) ] ;
                        gIn = (float)input2[(int)(3*BlockDims.x * (yindIn) + 3*(xindIn) + 1)];
                        rIn = (float)input2[(int)(3*BlockDims.x * (yindIn) + 3*(xindIn) + 2)];
                    }
                    else{
                        bIn = (float)input[(int)(3*BlockDims.x * (yindIn) + 3*(xindIn)) ] ;
                        gIn = (float)input[(int)(3*BlockDims.x * (yindIn) + 3*(xindIn) + 1)];
                        rIn = (float)input[(int)(3*BlockDims.x * (yindIn) + 3*(xindIn) + 2)];
                    
                    }

                    
                    

                    
                    if(boolhueshift&&(GrainMask[g].z==1.0)){
                       // keep ang value
                    }
                    else{
                        Ang =0;
                    }
               
                    float S;
                    if(boolSatChange&&(GrainMask[g].z==1.0)){
                      S = 2.1*SLD4/127.0;
                    }
                    else{
                        S = 1.0;
                    }
                    float SU = S*cos(Ang);
                    float SW = S*sin(Ang);
                    float rMed,gMed,bMed;
                    
                  
                    rMed = (.299+.701*SU+.168*SW)*rIn
                    + (.587-.587*SU+.330*SW)*gIn
                    + (.114-.114*SU-.497*SW)*bIn;
                    gMed = (.299-.299*SU-.328*SW)*rIn
                    + (.587+.413*SU+.035*SW)*gIn
                    + (.114-.114*SU+.292*SW)*bIn;
                    bMed = (.299-.3*SU+1.25*SW)*rIn
                    + (.587-.588*SU-1.05*SW)*gIn
                    + (.114+.886*SU-.203*SW)*bIn;
                   
                    
                    // from:
                    // http://beesbuzz.biz/code/hsv_color_transforms.php
                    
                    
                    
                    
                    
                    
                    
                    //pixels at the output
                    // output indexes
                    
                    int xindOut;
                    int yindOut;
                    xindOut = x + OutGrains[g].x ;
                    yindOut = y + OutGrains[g].y;
                    

                    
                    if((boolShiftGrain)&&(GrainMask[g].z==1.0)){
                        xindOut = x + OutGrains[g].x + shiftAmount*Gs*cos(randAng[g]);
                        yindOut = y + OutGrains[g].y + shiftAmount*Gs*sin(randAng[g]);
                    }

                  
                    xindOut = (xindOut<0)?0:xindOut;
                    xindOut = (xindOut>outXsize-1)?outXsize-1:xindOut;
                    
                    yindOut = (yindOut<0)?0:yindOut;
                    yindOut = (yindOut>outYsize-1)?outYsize-1:yindOut;
                    
                    
                    
                    bOut = (float)output[3*CurrentOutput.cols * (yindOut) + 3*(xindOut) ] ;
                    gOut = (float)output[3*CurrentOutput.cols * (yindOut) + 3*(xindOut) + 1];
                    rOut = (float)output[3*CurrentOutput.cols * (yindOut) + 3*(xindOut) + 2];
                    

                    output[3*CurrentOutput.cols * (yindOut) + 3*(xindOut) ] = (uchar)(bOut + Scale*(1.0 - boolShiftGrain*.5)*bMed);
                    output[3*CurrentOutput.cols * (yindOut) + 3*(xindOut) + 1]=(uchar)(gOut +Scale*(1.0 - boolShiftGrain*.5)*gMed);
                    output[3*CurrentOutput.cols * (yindOut) + 3*(xindOut) + 2]=(uchar)(rOut + Scale*(1.0 - boolShiftGrain*.5)*rMed);
                    
                    
                
                
                } //y
            } //x
            
        }//m
        
        if ((fc%jp) == jp-1) {
            FirstGrains[g] = SecondGrains[g];
        }
        
        
        
    }//g
    
    }
    
}




//--------------------------------------------------------------
void testApp::draw(){
	ofSetHexColor(0xffffff);

    
    ofxCvColorImage AuxDrawImage2;
       
    AuxDrawImage2.allocate(CurrentOutput.cols, CurrentOutput.rows);
    AuxDrawImage2 = CurrentOutput.data;
    
    AuxDrawImage2.draw(0, 0, ofGetWidth(),ofGetHeight());
   

    fc++;
    if(fc>BlockDims.z-1){fc=0;}
    
}


void testApp::exit(){

  
 //   vidGrabber1.close();
   // vidGrabber2.close();


}

void InitGrains(){
// calculates the grains input and output positions.
    FirstGrains.clear();
    SecondGrains.clear();
    OutGrains.clear();
    
    // input jump
    int Osamp = (int)(ceil(Gs*inOp/100.0));
    int jp = Gs - Osamp;
    
    // output jump
    int outOsamp = (int)(ceil(Gs*outOp/100.0));
    int ojp = Gs - outOsamp;
    
    int graincounter = 0;
    
    for (int y = 0; y < (BlockDims.y - Gs); y+=jp) {
        for (int x=0; x< (BlockDims.x - Gs); x+=jp) {
                ofVec3f tempStorage;
                tempStorage.x = x;
                tempStorage.y=y;
                tempStorage.z=0;
                FirstGrains.push_back(tempStorage);
                SecondGrains.push_back(tempStorage);
                OutGrains.push_back(tempStorage);
                GrainMask.push_back(tempStorage);
                randAng.push_back(ofRandom(2*PI));
                GrainsRandom.push_back(graincounter);
            graincounter++;
        }
    }

    // shufling the random grain array:
    
    for (int k = graincounter-1 ; k >= 0; k--) {
        int indi = (int)ofRandom(k);
        int tempo = GrainsRandom[k];
        GrainsRandom[k] = GrainsRandom[indi];
        GrainsRandom[indi] = tempo;
        cout<<"k: "<<k<<", indi: "<< GrainsRandom[k]<<endl;
    }
    
    
    
}




float CalculateScalecorrect(void){

    vector<float> Tempovec(10*Gs);
    // initializing
    for (int k=0; k<(10*Gs); k++) {
        Tempovec[k]=0.0;
    }
    
    // output jump
    int outOsamp = (int)(ceil(Gs*outOp/100.0));
    int ojp = Gs - outOsamp;
    
    // overlaping
    for (int k=0; k<(9*Gs); k+=ojp) {
        for (int n=0; n<Gs; n++) {
            Tempovec[k+n]+= 0.5*(1.0 -cosf(2*PI*n/(float)(Gs-1)));
        }
    }

    // finding the max;
    float TheMax =0;
    for (int k=0; k<(10*Gs); k++) {
        if(Tempovec[k]>TheMax){
            TheMax = Tempovec[k];
        }
    }

    return TheMax;

}

// Face detection function

/** @function detectAndDisplay */
void testApp::detectFace( cv::Mat frame )
{
    std::vector<cv::Rect> faces;
    cv::Mat frame_gray;
    
    cv::cvtColor( frame, frame_gray, CV_BGR2GRAY );
    cv::equalizeHist( frame_gray, frame_gray );
    
    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(60, 60) );
    
    //-- Detect faces
    //    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0, cv::Size(30, 30));
    
    //    cout<<faces.size()<<endl;
    for( int i = 0; i < faces.size(); i++ )
    {
         // Saving th face number 1 for recentering.
        
        if (i==0){
            FirstFace = faces[0];
        }
        
      }
    
}



    
    
    
    
    
    
    


// Motion detection function
// It marks as active the grains above a threshold
// The number of previous frames to be evaluated is an input parameter.


void testApp::detectMotion(int Nprev){
    
    
    // the current image:
    float ThMotion = KNOBMOTION/2.0;
    cv::Mat Graycurr;
    cv::cvtColor(TheFramesInput[fc], Graycurr, CV_RGB2GRAY);
    cv::Mat Acumm =cv::Mat::zeros(Graycurr.rows,Graycurr.cols, CV_8UC1);

    for (int k =0; k < Nprev; k++) {
        
        // reading a frame in the circular buffer:
        
        int LocalIndi = fc+ k +1;
        if(LocalIndi >= Nlast){LocalIndi = 0;}
        cv::Mat AuxGray;
        cv::cvtColor(TheFramesInput[LocalIndi], AuxGray, CV_RGB2GRAY);
        
        // calculating the absolute diferences with the next one
        cv::Mat TempoDiff;
        cv::absdiff(Graycurr, AuxGray, TempoDiff);
        // acumulating differences
        cv::add(Acumm, TempoDiff, Acumm);
        // updating the frame
        AuxGray.copyTo(Graycurr);
     }
    
    // Turning on grains depending on  changes:
    
    for (int g =0; g < FirstGrains.size(); g++) {
    
        unsigned char *inputAcumm = (unsigned char*)(Acumm.data);
        
        float GrainDiffSum = 0.0;
        int xindAcum,yindAcum;
        for (int x =0; x<Gs; x++) {
            for (int y=0; y<Gs; y++) {
                
                xindAcum = x + FirstGrains[g].x;
                yindAcum = y + FirstGrains[g].y;
                
                GrainDiffSum += (float)inputAcumm[Acumm.cols * (yindAcum) + (xindAcum) ];
            
            }
        }
    
        if (GrainDiffSum > Gs*Gs*ThMotion) {
            
             GrainMask[g].z = 1.0;
            
        }
        else{
        
             GrainMask[g].z = 0.0;
        }
    
    
    }
    
        

}








//--------------------------------------------------------------
void testApp::keyPressed(int key){
    
    switch (key) {
        case 'u':
            cout<<SLD7<<endl;

            break;
            
        case 'r':
         

            
            break;
        case OF_KEY_RIGHT:
           
            ModeSelector++;
            if(ModeSelector>12){ModeSelector =12;}
            if (ModeSelector ==6){randomReady=false;}
            
            break;
        case OF_KEY_LEFT:
            ModeSelector--;
            if (ModeSelector <1){ModeSelector =1;}
            if (ModeSelector ==6){randomReady=false;}
            
            break;
        case 't':
 

            break;
            
        case OF_KEY_RETURN:
   
    
           

            break;
        case 'q':
          
            break;
        default:
            break;
    }
    
    
}


//
void testApp::newMidiMessage(ofxMidiMessage& msg) {
    
	
	switch(msg.control){
            // slidders:
            case 0: // slidder 1 rotation
                SLD1 = msg .value;
            cout<<"SLD1 = "<<SLD1<<";"<<endl;
            break;
            case 1: // slidder 2 time delay
                SLD2 = msg .value;
            cout<<"SLD2 = "<<SLD2<<";"<<endl;
            break;
            case 2: // slidder 3 hue
            SLD3 = msg .value;
            cout<<"SLD3 = "<<SLD3<<";"<<endl;
            break;
            case 3: // slidder 4 sat
            SLD4 = msg .value;
            cout<<"SLD4 = "<<SLD4<<";"<<endl;
            break;
            case 4: // slidder 4 sat
            SLD5 = msg .value;
            cout<<"SLD5 = "<<SLD5<<";"<<endl;
            break;
            case 5:
            SLD6 = msg .value;
            cout<<"SLD6 = "<<SLD6<<";"<<endl;
            break;
            case 6:
            SLD7 = msg .value;
            cout<<"SLD7 = "<<SLD7<<";"<<endl;
            break;
            case 7:
            SLD8 = msg .value;
            cout<<"SLD8 = "<<SLD8<<";"<<endl;
            break;         
            
            //KNOBS
            case 23:
            KNOBMOTION = msg .value;
            cout<<"KNOB = "<<KNOBMOTION<<";"<<endl;
            break; 
            
            
            // Buttons:


            case 64:
          
            break;
            case 48:
            
            break;
            case 43:
            if(msg.value >100){
                ModeSelector--;
                if (ModeSelector <1){ModeSelector =1;}
                if (ModeSelector ==6){randomReady=false;}
            }
            break;
         case 44:
            if(msg.value >100){
                ModeSelector++;
                if(ModeSelector>12){ModeSelector =12;}
                if (ModeSelector ==6){randomReady=false;}
            }
            break;
           case 49:
            if(msg.value >100){


            }
            break;
            
        case 50:
            
            break;
        case 66:
            if(msg.value >100){
              
            }
            break;
        case 51:
          
            break;
        case 67:
            if(msg.value >100){
               
            }
            break;
        case 53:
          
            break;
        case 69:
            if(msg.value >100){
                
            }
            break;
            
            
        case 60:
            if(msg.value >100){
            
            }
            break;

        case 52:
            
            break;
        case 68:
            if(msg.value >100){
               
            }
            break;
            
            

    }
    
}

// Audio callback

void testApp::audioIn(float *Ainput, int BufferSize, int nChannels){

// samples are interleaved
    float SampleAverage =0;
    float RMSValue = 0;
    for (int i = 0; i < BuffS; i++) {
        if (nChannels ==2){
        SampleAverage = Ainput[i*2]*0.5 + Ainput[i*2+1]*0.5;
        }
        else if (nChannels ==1){
            SampleAverage = Ainput[i];
        }
            //AudioValues[i] = SampleAverage;
        RMSValue += SampleAverage*SampleAverage;
    }
    CurrentEnergy = 0.0;
    EnergyBuffer[EnergyIndy] = RMSValue/BuffS;
    EnergyIndy++;
    if (EnergyIndy==4){EnergyIndy=0;}
    for (int k = 0; k<4; k++) {
        CurrentEnergy+=(SLD8/32.0*SLD8/32.0)*EnergyBuffer[k];
    }
    
    
}



// Audio based selections

void testApp::EnergyCenterSelect(){

    
    float centerX = BlockDims.x/2.0;
    float centerY = BlockDims.y/2.0;
    //cout<<"Cx: "<<centerX<<", Cy: "<<centerY<<endl;
    float ThAudio = 0.1;
    for (int g=0; g < GrainMask.size(); g++) {
    
        float Xpos = GrainMask[g].x+Gs/2.0;
        float Ypos = GrainMask[g].y+Gs/2.0;
        
        float FarMeasure = 1.0 - 2*sqrtf((Xpos-centerX)*(Xpos-centerX)
                                         +(Ypos-centerY)*(Ypos-centerY))/sqrtf(Nx*Nx+Ny*Ny);
        
       //cout <<FarMeasure*CurrentEnergy<<endl;
        if (SLD7/127.0*FarMeasure*sqrtf(CurrentEnergy)> ThAudio){
            GrainMask[g].z = 1.0;
          }
        else{
            GrainMask[g].z = 0.0;
        }
    
    }
}

void testApp::AudioRandomSelect(){
    

    for (int g=0; g < GrainMask.size(); g++) {
        
  
        if (g < 25*SLD7/127.0*GrainMask.size()*sqrtf(CurrentEnergy)){
            GrainMask[GrainsRandom[g]].z = 1.0;
        }
        else{
            GrainMask[GrainsRandom[g]].z = 0.0;
        }
        
    }
    
}





//--------------------------------------------------------------
void testApp::keyReleased(int key){
    
}

//--------------------------------------------------------------
void testApp::mouseMoved(int x, int y){
    
}

//--------------------------------------------------------------
void testApp::mouseDragged(int x, int y, int button){
    
}

//--------------------------------------------------------------
void testApp::mousePressed(int x, int y, int button){
    
}

//--------------------------------------------------------------
void testApp::mouseReleased(int x, int y, int button){
    
}

//--------------------------------------------------------------
void testApp::windowResized(int w, int h){
    
}

//--------------------------------------------------------------
void testApp::gotMessage(ofMessage msg){
    
}

//--------------------------------------------------------------
void testApp::dragEvent(ofDragInfo dragInfo){ 
    
}