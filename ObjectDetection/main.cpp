//
//  main.cpp
//  ObjectDetection
//
//  Created by 서정 on 2021/07/01.
//

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void detectFaceHand(string video);
void detectVehicle(string video);

typedef struct box {
    int top;
    int left;
    int bottom;
    int right;
} Box;

int memory[1000];
int handmemory[1000];
Box bb[1000];
Box bb2[1000];


int find (int x, int* arr) {
    if(arr[x] == x) {
        return x;
    } else {
        return arr[x] = find(arr[x], arr);
    }
}

void Union(int x, int y, int* arr) {
    x = find(x, arr);
    y = find(y, arr);

    arr[y] = x;
}

int main(int argc, const char * argv[]) {
    // insert code here...
    
    //hand
    string video_name_1 = "/Users/sjng/Documents/ObjectDetection/ObjectDetection/Hand Video2.mov";
    string video_name_2 = "/Users/sjng/Documents/ObjectDetection/ObjectDetection/Project- hand gesture.AVI";
    
    //car
    string video_name_3 = "/Users/sjng/Documents/ObjectDetection/ObjectDetection/Project_outdoor video1.mov";
    string video_name_4 = "/Users/sjng/Documents/ObjectDetection/ObjectDetection/Car video2.mp4";
    
    //detectFaceHand(video_name_2);
    detectVehicle(video_name_4);
    
    return 0;
}

void detectFaceHand(string video) {
    
    VideoCapture cap;
    cap.open(video);
    Mat frame1, prevImage;
    Mat bgr, gray;
    
    cap >> frame1;
    cvtColor(frame1, prevImage, COLOR_BGR2GRAY);
    
    int cnt = 0;
    for( ; ; ) {
        Mat frame, currImage;
        cap >> frame;
        cnt++;
        if(frame.data == nullptr) break;
        
        Mat th_skin(frame.size(), 0);
        Mat hsvImage;
        cvtColor(frame, hsvImage, COLOR_BGR2HSV);
        cvtColor(frame, currImage, COLOR_BGR2GRAY);
        
        Mat handImage = Mat::zeros(frame.size(), 0);
        Mat labelImage = Mat::zeros(frame.size(),CV_32S);
        Mat handLabelImage = Mat::zeros(frame.size(),CV_32S);
        
        
        if(cnt % 1 == 0){
            
            // Thresholding by skin color
            for(int i=0; i<frame.size().height; i++) {
                for (int j=0; j<frame.size().width; j++) {
                    uchar h = hsvImage.at<Vec3b>(i,j)[0];
                    uchar s = hsvImage.at<Vec3b>(i,j)[1];
                    uchar v = hsvImage.at<Vec3b>(i,j)[2];
                    
                    if(h>=0 && h<=20 && s>=48 && v>=60){
                        th_skin.at<uchar>(i,j) = 255;
                    } else {
                        th_skin.at<uchar>(i,j) = 0;
                    }
                }
            }
            
            // Preprocessing: erosion, dilation
            erode(th_skin, th_skin, Mat::ones(Size(3, 3), CV_8UC1), Point(-1, -1), 3);
            dilate(th_skin, th_skin, Mat::ones(Size(3, 3), CV_8UC1), Point(-1, -1), 4);
            
            
            
            // Motion detection
            Mat flow;
            calcOpticalFlowFarneback(prevImage, currImage, flow, .5, 3, 15, 3, 4, 1.2, 0);
            
            // Motion detection -> visualization
            Mat flow_parts[2];
            split(flow, flow_parts);
            Mat magnitude, angle, magn_norm;
            cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
            normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
            angle *= ((1.f / 360.f) * (180.f / 255.f));
            
            //Motion detection -> build hsv image
            Mat _hsv[3], hsv, hsv8;
            _hsv[0] = angle;
            _hsv[1] = Mat::ones(angle.size(), CV_32F);
            _hsv[2] = magn_norm;
            merge(_hsv, 3, hsv);
            hsv.convertTo(hsv8, CV_8U, 255.0);
            cvtColor(hsv8, bgr, COLOR_HSV2BGR);
            cvtColor(bgr, gray, COLOR_BGR2GRAY);

            // ‘& operation’ between motion and skin color
            for(int i=0; i<th_skin.size().height; i++) {
                for (int j=0; j<th_skin.size().width; j++) {
                    if(th_skin.at<uchar>(i,j) & (gray.at<uchar>(i,j)>50))
                        handImage.at<uchar>(i,j) = 255;
                }
            }
            
            
            //Connected component labeling
            int num = 0;
            int handNum = 0;
            memset(memory, 0, sizeof(memory));
            memset(handmemory, 0, sizeof(handmemory));
            memset(bb, -1, sizeof(bb));
            memset(bb2, -1, sizeof(bb2));
            
            for(int i=1; i<th_skin.size().height; i++) {
                for (int j=1; j<th_skin.size().width; j++) {
                    
                    // Face
                    if(th_skin.at<uchar>(i,j) > 0) {
                        if((th_skin.at<uchar>(i-1, j) | th_skin.at<uchar>(i, j-1))==0) {
                            labelImage.at<int>(i,j) = ++num;
                            memory[labelImage.at<int>(i,j)] = labelImage.at<int>(i,j);
                        } else if((th_skin.at<uchar>(i-1, j) & th_skin.at<uchar>(i, j-1)) == 0) {
                            labelImage.at<int>(i,j) = find(max(labelImage.at<int>(i-1, j),  labelImage.at<int>(i, j-1)), memory);
                        } else {
                            labelImage.at<int>(i,j) = find(min(labelImage.at<int>(i-1, j),  labelImage.at<int>(i, j-1)),memory);
                            Union(labelImage.at<int>(i,j), max(labelImage.at<int>(i-1, j),  labelImage.at<int>(i, j-1)), memory);
                        }
                    }
                    
                    // Hand
                    if(handImage.at<uchar>(i,j) > 0) {
                        if((handImage.at<uchar>(i-1, j) | handImage.at<uchar>(i, j-1))==0) {
                            handLabelImage.at<int>(i,j) = ++handNum;
                            handmemory[handLabelImage.at<int>(i,j)] = handLabelImage.at<int>(i,j);
                        } else if((handImage.at<uchar>(i-1, j) & handImage.at<uchar>(i, j-1)) == 0) {
                            handLabelImage.at<int>(i,j) = find(max(handLabelImage.at<int>(i-1, j),  handLabelImage.at<int>(i, j-1)),handmemory);
                        } else {
                            handLabelImage.at<int>(i,j) = find(min(handLabelImage.at<int>(i-1, j),  handLabelImage.at<int>(i, j-1)),handmemory);
                            Union(handLabelImage.at<int>(i,j), max(handLabelImage.at<int>(i-1, j),  handLabelImage.at<int>(i, j-1)), handmemory);
                        }
                    }
                }
            }
            
            
            int maxarea = 1500;
            
            // Find box point
            for(int i=1; i<th_skin.size().height; i++) {
                for (int j=1; j<th_skin.size().width; j++) {
                    
                    // Face
                    if(labelImage.at<int>(i,j)!=0) {
                        labelImage.at<int>(i,j) = find(labelImage.at<int>(i,j), memory);
                        if(bb[labelImage.at<int>(i,j)].top == -1 || bb[labelImage.at<int>(i,j)].top > i) {
                            bb[labelImage.at<int>(i,j)].top = i;
                        }

                        if(bb[labelImage.at<int>(i,j)].left == -1 || bb[labelImage.at<int>(i,j)].left > j) {
                            bb[labelImage.at<int>(i,j)].left = j;
                        }

                        if(bb[labelImage.at<int>(i,j)].bottom == -1 || bb[labelImage.at<int>(i,j)].bottom < i) {
                            bb[labelImage.at<int>(i,j)].bottom = i;
                        }

                        if(bb[labelImage.at<int>(i,j)].right == -1 || bb[labelImage.at<int>(i,j)].right < j) {
                            bb[labelImage.at<int>(i,j)].right = j;
                        }
                        
                    }
                    
                    // Hand
                    if(handLabelImage.at<int>(i,j)!=0) {
                        handLabelImage.at<int>(i,j) = find(handLabelImage.at<int>(i,j), handmemory);
                        if(bb2[handLabelImage.at<int>(i,j)].top == -1 || bb2[handLabelImage.at<int>(i,j)].top > i) {
                            bb2[handLabelImage.at<int>(i,j)].top = i;
                        }

                        if(bb2[handLabelImage.at<int>(i,j)].left == -1 || bb2[handLabelImage.at<int>(i,j)].left > j) {
                            bb2[handLabelImage.at<int>(i,j)].left = j;
                        }

                        if(bb2[handLabelImage.at<int>(i,j)].bottom == -1 || bb2[handLabelImage.at<int>(i,j)].bottom < i) {
                            bb2[handLabelImage.at<int>(i,j)].bottom = i;
                        }

                        if(bb2[handLabelImage.at<int>(i,j)].right == -1 || bb2[handLabelImage.at<int>(i,j)].right < j) {
                            bb2[handLabelImage.at<int>(i,j)].right = j;
                        }
                        
                    }
                }
            }
            
            // Find area not too small
            for(int i=0; i<1000; i++) {
                
                // Face
                if(bb[i].top!=-1 && (bb[i].right-bb[i].left) * (bb[i].bottom-bb[i].top)>=maxarea) {
                    
                    for(int a=bb[i].left; a<bb[i].right; a++) {
                        frame.at<Vec3b>(bb[i].top, a) = 255;
                        frame.at<Vec3b>(bb[i].bottom, a) = 255;

                    }
                    for (int b=bb[i].top; b<bb[i].bottom; b++) {
                        frame.at<Vec3b>(b, bb[i].left) = 255;
                        frame.at<Vec3b>(b, bb[i].right) = 255;
                    }
                }
                // Hand
                if(bb2[i].top!=-1 && (bb2[i].right-bb2[i].left) * (bb2[i].bottom-bb2[i].top) > 100) {

                    for(int a=bb2[i].left; a<bb2[i].right; a++) {
                        frame.at<Vec3b>(bb2[i].top, a) = {0,0,255};
                        frame.at<Vec3b>(bb2[i].bottom, a) = {0,0,255};

                    }
                    for (int b=bb2[i].top; b<bb2[i].bottom; b++) {
                        frame.at<Vec3b>(b, bb2[i].left) = {0,0,255};
                        frame.at<Vec3b>(b, bb2[i].right) = {0,0,255};
                    }
                }
            }
            
            prevImage = currImage;
        
            
        }
        
        
        imshow("video", frame);
        if(waitKey(30)>=0) break;
        
    }
    
}


void detectVehicle(string video) {
    
    VideoCapture cap;
    cap.open(video);
    int cnt = 0;
    Mat frame1;
    cap >> frame1;
    Mat avgImage(frame1.size(), CV_32FC1, Scalar());
    Mat bgImage;
    
    
    for ( ; ; ) {
        Mat frame;
        cap >> frame;
        if(frame.data == nullptr) break;
        
        
        Mat grayFrame;
        
        // Preprocessing
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
        equalizeHist(grayFrame, grayFrame);
        
        Mat thImage = grayFrame;
        accumulate(grayFrame, avgImage);
        
        Mat labelImage = Mat::zeros(frame.size(),CV_32S);
        
        
        // Detect background image (Averaging)
        if(++cnt % 1000 == 0) {
            avgImage.convertTo(bgImage, CV_8UC3, 1.0/1000);
            avgImage.zeros(frame.size(), CV_32FC3);
            //imshow("avg", bgImage);
        }
        
        if(bgImage.size() != Size(0,0)) {
            
            // Subtract background from image
            for(int i=0; i<grayFrame.size().height; i++) {
                for(int j=0; j<grayFrame.size().width; j++) {
                    grayFrame.at<uchar>(i,j) = max(grayFrame.at<uchar>(i,j), bgImage.at<uchar>(i,j)) - min(grayFrame.at<uchar>(i,j), bgImage.at<uchar>(i,j));
                }
            }
            
            // Thresholding
            for(int i=0; i<thImage.size().height; i++) {
                for(int j=0; j<thImage.size().width; j++) {
                    if(thImage.at<uchar>(i,j)>60) {
                        thImage.at<uchar>(i,j) = 255;
                    } else thImage.at<uchar>(i,j) = 0;
                }
            }
            
            // erosion and dilation
            erode(thImage, thImage, Mat::ones(Size(3, 3), CV_8UC1), Point(-1, -1), 1);
            dilate(thImage, thImage, Mat::ones(Size(3, 3), CV_8UC1), Point(-1, -1), 1);
            
            // Connected component labeling
            int num = 0;
            memset(memory, 0, sizeof(memory));
            memset(bb, -1, sizeof(bb));
            for(int i=1; i<thImage.size().height; i++) {
                for(int j=1; j<thImage.size().width; j++) {
                    if(thImage.at<uchar>(i,j) > 0) {
                        if((thImage.at<uchar>(i-1, j) | thImage.at<uchar>(i, j-1))==0) {
                            labelImage.at<int>(i,j) = ++num;
                            memory[labelImage.at<int>(i,j)] = labelImage.at<int>(i,j);
                        } else if((thImage.at<uchar>(i-1, j) & thImage.at<uchar>(i, j-1)) == 0) {
                            labelImage.at<int>(i,j) = max(labelImage.at<int>(i-1, j),  labelImage.at<int>(i, j-1));
                        } else {
                            labelImage.at<int>(i,j) = min(labelImage.at<int>(i-1, j),  labelImage.at<int>(i, j-1));
                            Union(labelImage.at<int>(i,j), max(labelImage.at<int>(i-1, j),  labelImage.at<int>(i, j-1)), memory);
                        }
                    }
                }
            }

            // Find box point
            for(int i=1; i<thImage.size().height; i++) {
                for (int j=1; j<thImage.size().width; j++) {
                    if(labelImage.at<int>(i,j)!=0) {
                        labelImage.at<int>(i,j) = find(labelImage.at<int>(i,j), memory);

                        if(bb[labelImage.at<int>(i,j)].top == -1 || bb[labelImage.at<int>(i,j)].top > i) {
                            bb[labelImage.at<int>(i,j)].top = i;
                        }

                        if(bb[labelImage.at<int>(i,j)].left == -1 || bb[labelImage.at<int>(i,j)].left > j) {
                            bb[labelImage.at<int>(i,j)].left = j;
                        }

                        if(bb[labelImage.at<int>(i,j)].bottom == -1 || bb[labelImage.at<int>(i,j)].bottom < i) {
                            bb[labelImage.at<int>(i,j)].bottom = i;
                        }

                        if(bb[labelImage.at<int>(i,j)].right == -1 || bb[labelImage.at<int>(i,j)].right < j) {
                            bb[labelImage.at<int>(i,j)].right = j;
                        }
                    }
                }
            }

            //Find area not too small
            for(int i=0; i<1000; i++) {
                if(bb[i].top!=-1 && (bb[i].right-bb[i].left) * (bb[i].bottom-bb[i].top) > 1000) {

                    for(int a=bb[i].left; a<bb[i].right; a++) {
                        frame.at<Vec3b>(bb[i].top, a) = 255;
                        frame.at<Vec3b>(bb[i].bottom, a) = 255;

                    }
                    for (int b=bb[i].top; b<bb[i].bottom; b++) {
                        frame.at<Vec3b>(b, bb[i].left) = 255;
                        frame.at<Vec3b>(b, bb[i].right) = 255;
                    }
                }
            }
            
        }
        
        
        
        imshow("video2", frame);
        if(waitKey(30)>=0) break;
        
    }
    
}


