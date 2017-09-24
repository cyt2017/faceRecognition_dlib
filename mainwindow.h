#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <QImage>
#include <QTimer>
#include <QFile>
#include <QIODevice>
#include <QDir>
#include <QMessageBox>
#include<opencv2/ml/ml.hpp>
#include <opencv2/face.hpp>

using namespace cv::face;
using namespace dlib;
using namespace cv;
using namespace std;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    bool FaceDetect(Mat inSrc,Mat &faceRgb);
    void Dlib_Predefine();//dlib 预定义的函数

    bool getFaceAngle(Mat inSrc, double &angle);
    Mat faceRotate(Mat inSrc,double angle);
    bool getFace(Mat inSrc,Mat &rgbFace);
    bool getFaceSevenPoints(Mat inSrc,std::vector<Point> &points);

    void showImage_FacesRecognitionDemo(Mat inSrc);
    void saveImageText_FacesRecognitionDemo(QString imgPath);

    //svm
    Mat _mean;
    Mat _eigenvectors;
    void trainSVM_FacesRecognitionDemo();
    std::vector<Mat> getEigenFace(InputArrayOfArrays _src, InputArray _local_labels);
    int SVMpredict(Mat in);

protected slots:
    void onCaptureBtn_FacesRecognitionDemo();
    void onSaveBtn_FacesRecognitionDemo();
    void onTimeOut_FacesRecognitionDemo();
    void onTrainBtn_FacesRecognitionDemo();
    void onRecognitionBtn_FacesRecognitionDemo();
    void onTrainSvm();

private:
    Ui::MainWindow *ui;
    VideoCapture capture;
    Mat frame,result;
    QTimer *timer;
    int index;
    //bool bo_file;
    //frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor sp;//Already get
    dlib::frontal_face_detector detector;

    //recognition model
    Ptr<FaceRecognizer> model_E;
    Ptr<FaceRecognizer> model_F;
    Ptr<FaceRecognizer> model_L;
};

#endif // MAINWINDOW_H
