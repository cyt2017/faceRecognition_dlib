#include "mainwindow.h"
#include "ui_mainwindow.h"

const double mean_face_shape_x[] = {
    0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483, 0.799124,
    0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127, 0.36688, 0.426036,
    0.490127, 0.554217, 0.613373, 0.121737, 0.187122, 0.265825, 0.334606, 0.260918,
    0.182743, 0.645647, 0.714428, 0.793132, 0.858516, 0.79751, 0.719335, 0.254149,
    0.340985, 0.428858, 0.490127, 0.551395, 0.639268, 0.726104, 0.642159, 0.556721,
    0.490127, 0.423532, 0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874,
    0.553364, 0.490127, 0.42689
};
const double mean_face_shape_y[] = {
    0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
    0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625, 0.587326,
    0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758, 0.179852, 0.231733,
    0.245099, 0.244077, 0.231733, 0.179852, 0.178758, 0.216423, 0.244077, 0.245099,
    0.780233, 0.745405, 0.727388, 0.742578, 0.727388, 0.745405, 0.780233, 0.864805,
    0.902192, 0.909281, 0.902192, 0.864805, 0.784792, 0.778746, 0.785343, 0.778746,
    0.784792, 0.824182, 0.831803, 0.824182
};


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    Dlib_Predefine();
    if(!capture.open(0))
    {
        printf("capture.open() fail..\n");
    }
    timer = new QTimer(this);
    index = 0;


    ui->saveBtn->setEnabled(false);
    connect(timer,SIGNAL(timeout()),this,SLOT(onTimeOut_FacesRecognitionDemo()));
    timer->start(30);
    connect(ui->captureBtn,SIGNAL(clicked(bool)),this,SLOT(onCaptureBtn_FacesRecognitionDemo()));
    connect(ui->trainBtn,SIGNAL(clicked(bool)),this,SLOT(onTrainBtn_FacesRecognitionDemo()));
    connect(ui->recognitionBtn,SIGNAL(clicked(bool)),this,SLOT(onRecognitionBtn_FacesRecognitionDemo()));
    connect(ui->saveBtn,SIGNAL(clicked(bool)),this,SLOT(onSaveBtn_FacesRecognitionDemo()));
    connect(ui->trainSVMBtn,SIGNAL(clicked(bool)),this,SLOT(onTrainSvm()));

    detector = dlib::get_frontal_face_detector();

    //recognition model init
    model_E = cv::face::createEigenFaceRecognizer();
    model_E->load("model.xml");

    model_F = cv::face::createFisherFaceRecognizer();
    model_F->load("fisherFace_model.xml");

    model_L = cv::face::createLBPHFaceRecognizer();
    model_L->load("LBPHFace_model.xml");
}

MainWindow::~MainWindow()
{
    delete ui;
}

bool MainWindow::FaceDetect(Mat inSrc , Mat &faceRgb)
{
    Mat src,dst;
    double angle;
    src = inSrc.clone();
    if(!getFaceAngle(src,angle))
    {
       return false;
    }

    dst = faceRotate(src,angle);
    if(!getFace(dst,faceRgb))
    {
        return false;
    }
    return true;
}

void MainWindow::Dlib_Predefine()
{
    dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> sp;//读入标记点文件
}

bool MainWindow::getFaceAngle(Mat inSrc,double &angle)
{
    Mat gray;
    cv::cvtColor(inSrc, gray, CV_BGR2GRAY);

    dlib::cv_image<dlib::bgr_pixel> cimg(inSrc);
    std::vector<dlib::rectangle> faces = detector(cimg);
    int peopleNUM = (faces.size()>0 ? faces.size() : 0);
    if(peopleNUM!=1)
    {
        return false;
    }
    std::vector<dpoint> from_points, to_points;

    //人脸对齐技术提高了准确率
    dlib::full_object_detection shape = sp(cimg, faces[0]);//标记点
    double x1 = shape.part(27).x()-shape.part(0).x();
    double x2 = shape.part(16).x()-shape.part(27).x();
    if(x1/x2>2.0 || x1/x2<0.5)
    {
        return false;
    }

    for (int j = 0; j < 68; j++)
    {
//           //用来画特征值的点
        if(j==33 || j==36 || j==39 || j==42 || j==45 || j==48 || j==54)
        {
            dpoint p;
            p.x() = (mean_face_shape_x[j-17]);
            p.y() = (mean_face_shape_y[j-17]);
            from_points.push_back(p*200);
            to_points.push_back(shape.part(j));
        }
    }
    const point_transform_affine tform = find_similarity_transform(from_points,to_points);
    dlib::vector<double,2> p(1,0);
    p = tform.get_m()*p;
    angle = std::atan2(p.y(),p.x());
    angle = angle*180.0/CV_PI;
    return true;
}

Mat MainWindow::faceRotate(Mat inSrc, double angle)
{
    Mat src,dst;
    src = inSrc.clone();
    cv::Point2f center(src.cols / 2, src.rows / 2);
    cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1);
    cv::Rect bbox = cv::RotatedRect(center, src.size(), angle).boundingRect();

    rot.at<double>(0, 2) += bbox.width / 2.0 - center.x;
    rot.at<double>(1, 2) += bbox.height / 2.0 - center.y;
    cv::warpAffine(src, dst, rot, bbox.size());
    return dst;
}

bool MainWindow::getFace(Mat inSrc,Mat &rgbFace)
{
    Mat dst;
    dst = inSrc.clone();
//    Mat tmp = inSrc.clone();
    dlib::cv_image<dlib::bgr_pixel> cimg(dst);
    std::vector<dlib::rectangle> faces = detector(cimg);
    int peopleNUM = (faces.size()>0 ? faces.size() : 0);
    if(peopleNUM!=1)
    {
        return false;
    }
    cv::Rect cvrect = Rect(0,0,0,0);

    //人脸对齐技术提高了准确率
    dlib::full_object_detection shape = sp(cimg, faces[0]);//标记点
    std::vector<Point> points;
    for(int j=0;j<68;j++)
    {
        points.push_back(Point(shape.part(j).x(),shape.part(j).y()));
//        circle(tmp,points[j],1,Scalar(255,255,0));
    }
//    imshow("tmp",tmp);

    cvrect = cv::boundingRect(Mat(points));

    Mat pic;
    dst(cvrect).copyTo(pic);
//    cv::imshow("pic",pic);
//    cv::waitKey(-1);
    cv::cvtColor(pic, pic, CV_BGR2GRAY);
    rgbFace = pic.clone();
    return true;
}

bool MainWindow::getFaceSevenPoints(Mat inSrc, std::vector<Point> &points)
{
    dlib::cv_image<dlib::uint8> cimg(inSrc);
    dlib::rectangle face(0,0,inSrc.cols,inSrc.rows) ; //detector(cimg);
//    int peopleNUM = (faces.size()>0 ? faces.size() : 0);
//    if(peopleNUM!=1)
//    {
//        return false;
//    }


    //人脸对齐技术提高了准确率
    dlib::full_object_detection shape = sp(cimg, face);//标记点

    for (int j = 0; j < 68; j++)
    {
        //用来画特征值的点,
        //33鼻窝\ 36左眼左\  39左眼右\ 42右眼左\ 45右眼右\ 48嘴角左\ 54嘴角右
        if(j==33 || j==36 || j==39 || j==42 || j==45 || j==48 || j==54)
        {
            points.push_back(Point(shape.part(j).x(),shape.part(j).y()));
        }
    }
    return true;
}

void MainWindow::showImage_FacesRecognitionDemo(Mat inSrc)
{
    QImage img;
    Mat src = inSrc.clone();
    if(src.channels()==1)
    {
        img = QImage(src.data,src.cols,src.rows,src.cols,QImage::Format_Indexed8);
        ui->label_gray->setPixmap(QPixmap::fromImage(img));
    }
    else if(src.channels()==3)
    {
        img = QImage(src.data,src.cols,src.rows,src.cols*3,QImage::Format_RGB888);
        ui->label_org->setPixmap(QPixmap::fromImage(img));
    }
}

void MainWindow::saveImageText_FacesRecognitionDemo(QString imgPath)
{
    QString str;
    str = imgPath + str.sprintf(";%d\n",ui->comboBox->currentIndex()+1);
    QFile file("faceImgText.csv");
    if(!file.open(QIODevice::Append))
    {
        printf("file.open() fail..\n");
    }
    file.write(str.toLatin1().data(),str.length());
    file.close();
}

inline Mat asRowMatrix(InputArrayOfArrays src, int rtype, double alpha=1, double beta=0) {
    // make sure the input data is a vector of matrices or vector of vector
    if(src.kind() != _InputArray::STD_VECTOR_MAT && src.kind() != _InputArray::STD_VECTOR_VECTOR) {
        String error_message = "The data is expected as InputArray::STD_VECTOR_MAT (a std::vector<Mat>) or _InputArray::STD_VECTOR_VECTOR (a std::vector< std::vector<...> >).";
//        CV_Error(Error::StsBadArg, error_message);
    }
    // number of samples
    size_t n = src.total();
    // return empty matrix if no matrices given
    if(n == 0)
        return Mat();
    // dimensionality of (reshaped) samples
    size_t d = src.getMat(0).total();
    // create data matrix
    Mat data((int)n, (int)d, rtype);
    // now copy data
    for(unsigned int i = 0; i < n; i++) {
        // make sure data can be reshaped, throw exception if not!
        if(src.getMat(i).total() != d) {
            String error_message = format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, d, src.getMat(i).total());
//            CV_Error(Error::StsBadArg, error_message);
        }
        // get a hold of the current row
        Mat xi = data.row(i);
        // make reshape happy by cloning for non-continuous matrices
        if(src.getMat(i).isContinuous()) {
            src.getMat(i).reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        } else {
            src.getMat(i).clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        }
    }
    return data;
}
std::vector<Mat> MainWindow::getEigenFace(InputArrayOfArrays _src, InputArray _local_labels)
{
    std::vector<Mat>_projections;
    // get labels
        Mat labels = _local_labels.getMat();
        // observations in row
        Mat data = asRowMatrix(_src, CV_32FC1);

        // number of samples
       int n = data.rows;
        // assert there are as much samples as labels
        if(static_cast<int>(labels.total()) != n) {
            String error_message = format("The number of samples (src) must equal the number of labels (labels)! len(src)=%d, len(labels)=%d.", n, labels.total());
//            CV_Error(Error::StsBadArg, error_message);
        }
        // clear existing model data

        _projections.clear();
        // clip number of components to be valid
        int _num_components = n;

        // perform the PCA
        PCA pca(data, Mat(), cv::PCA::DATA_AS_ROW, _num_components);
        // copy the PCA results

        _mean = pca.mean.reshape(1,1); // store the mean vector
        Mat _eigenvalues = pca.eigenvalues.clone(); // eigenvalues by row
        transpose(pca.eigenvectors, _eigenvectors); // eigenvectors by column

        // save projections
        for(int sampleIdx = 0; sampleIdx < data.rows; sampleIdx++) {
            Mat p = LDA::subspaceProject(_eigenvectors, _mean, data.row(sampleIdx));
            _projections.push_back(p);

        }
        return _projections;
}

int MainWindow::SVMpredict(Mat in)
{
    Mat src;
    in.convertTo(src,CV_32FC1);
    Mat p = LDA::subspaceProject(_eigenvectors, _mean,src.reshape(0,1));
    printf("p:%d - %d\n",p.type(),src.type());
    printf("pcols:%d - %d\n",p.cols,src.cols);
    fflush(NULL);
    Ptr<cv::ml::SVM> model =cv::ml::SVM::load("svm_model.xml");

    printf("getVarCount:%d  \n",model->getVarCount());
    fflush(NULL);
    float label = model->predict(p);
    printf("label:%f\n",label);
    fflush(NULL);
    return (int)label;
}

void MainWindow::trainSVM_FacesRecognitionDemo()
{
//    Mat mean = model_E->getMat("mean");
//    Mat eigenvectors = model_E->getMat("eigenvectors");
//    Mat p = LDA::subspaceProject(eigenvectors, mean, data.row(sampleIdx));
    std::vector<Mat> images;
    std::vector<int> labels;
    QDir dir("imgGray/");
    if(!dir.exists())
    {
        printf("dir.exists() fail..\n");
    }
    dir.setFilter(QDir::Files|QDir::NoSymLinks);
    QFileInfoList lists = dir.entryInfoList();
    QString str;
    for(int i=0;i<lists.count();i++)
    {
        str = lists.at(i).absoluteFilePath();
        Mat src = cv::imread(str.toStdString(),0);
        if(src.empty())
        {
            continue;
        }
        images.push_back(src.clone());
        QString ss = str.mid(str.lastIndexOf("/")+1,str.lastIndexOf("_")-str.lastIndexOf("/")-1);
        labels.push_back(ss.toInt());
    }
    std::vector<Mat> eigenFace = getEigenFace(images, labels);
    Mat trainingDataMat = asRowMatrix(eigenFace, CV_32FC1);
    Mat trainingLabelMat=Mat(labels.size(),1,CV_32SC1);
    for(int i=0;i<labels.size();i++)
    {
        trainingLabelMat.at<int>(i,0) = labels[i];
    }

    // 创建分类器并设置参数
      Ptr<cv::ml::SVM> model =cv::ml::SVM::create();
      model->setType(cv::ml::SVM::C_SVC);
      model->setKernel(cv::ml::SVM::RBF);  //核函数
      model->setC(10.0);
      model->setTermCriteria(cv::TermCriteria(CV_TERMCRIT_EPS,1000,FLT_EPSILON));

      //设置训练数据
     Ptr<cv::ml::TrainData> tData =cv::ml::TrainData::create(trainingDataMat, cv::ml::ROW_SAMPLE, trainingLabelMat);
     // 训练分类器
     model->train(tData);
     model->save("svm_model.xml");
     QMessageBox::information(this,"title","model finish");
}

void MainWindow::onCaptureBtn_FacesRecognitionDemo()
{
    Mat gray;
    std::vector<Point> points;
    bool xR = false,yR = false;
    if(FaceDetect(frame.clone(),gray)==true)
    {
        if(!gray.empty())
        {
            cv::Size size;
            ui->saveBtn->setEnabled(true);
            cv::equalizeHist(gray, result);//直方图均衡
            float xRatio = 92.0/result.cols;
            float yRatio = 112.0/result.rows;
            if(xRatio>yRatio)
            {
                size.width = (int)result.cols*xRatio;
                size.height = (int)result.rows*xRatio;
                xR = true;
            }
            else
            {
                size.width = (int)result.cols*yRatio;
                size.height = (int)result.rows*yRatio;
                yR = true;
            }
            cv::resize(result, result, size);//裁剪
//            imshow("result",result);
//            waitKey(-1);
            if(getFaceSevenPoints(result,points))
            {/*
                for(int i=0;i<points.size();i++)
                {
                    printf("points %d : [%d,%d]\n",i,points[i].x,points[i].y);
                    fflush(NULL);
                }*/
                int x , y;
                if(xR)
                {
                    x = 0;
                    y = (int)(points[0].y-112/2);
                }
                if(yR)
                {
                    x = (int)((points[4].x-points[1].x)/2+points[1].x-92/2);
                    y=0;
                }
//                int x = (int)((points[4].x-points[1].x)/2+points[1].x-92/2);
//                int y = (int)(points[0].y-112/2);
                int w = 92;
                int h = 112;
                printf("%d,%d,%d,%d\n result:[%d,%d]\n",x,y,w,h,result.cols,result.rows);
                fflush(NULL);
                if(x<0)
                {
                    x =0;
                }else
                {
                   if(x+92>result.cols)
                   {
                       x -=  x+92-result.cols;
                   }
                }
                if(y<0)
                {
                    y = 0;
                }else
                {
                    if(y+112>result.rows)
                    {
                        y -=  y+112-result.rows;
                    }
                }
                result(Rect(x,y,w,h)).copyTo(result);

                showImage_FacesRecognitionDemo(result);
            }

//            imshow("111",result);
//            waitKey(-1);

        }
    }
}

void MainWindow::onSaveBtn_FacesRecognitionDemo()
{
    QString str ;
    QFile file;
    while(1)
    {
        str.sprintf("imgGray/%d_%d.jpg",ui->comboBox->currentIndex(),index);
        file.setFileName(str);
        if(!file.exists())
        {
            break;
        }
        index++;
    }
    imwrite(str.toStdString(),result);

    ui->saveBtn->setEnabled(false);
}

void MainWindow::onTimeOut_FacesRecognitionDemo()
{
    while(!capture.read(frame));

    onRecognitionBtn_FacesRecognitionDemo();
    showImage_FacesRecognitionDemo(frame);
}

void MainWindow::onTrainBtn_FacesRecognitionDemo()
{
    std::vector<Mat> images;
    std::vector<int> labels;
    QDir dir("imgGray/");
    if(!dir.exists())
    {
        printf("dir.exists() fail..\n");
    }
    dir.setFilter(QDir::Files|QDir::NoSymLinks);
    QFileInfoList lists = dir.entryInfoList();
    QString str;
    for(int i=0;i<lists.count();i++)
    {
        str = lists.at(i).absoluteFilePath();
        Mat src = imread(str.toStdString(),0);
        if(src.empty())
        {
            continue;
        }
        images.push_back(src.clone());
        QString ss = str.mid(str.lastIndexOf("/")+1,str.lastIndexOf("_")-str.lastIndexOf("/")-1);
        labels.push_back(ss.toInt());
    }

    Ptr<FaceRecognizer> model;
    if(ui->comboBox_2->currentIndex()==0)
    {
        model = cv::face::createEigenFaceRecognizer();
        model->train(images, labels);
        model->save("model.xml");
    }
    else if(ui->comboBox_2->currentIndex()==1)
    {
        model = cv::face::createFisherFaceRecognizer();
        model->train(images, labels);
        model->save("fisherFace_model.xml");
    }
    else if(ui->comboBox_2->currentIndex()==2)
    {
        model = cv::face::createLBPHFaceRecognizer();
        model->train(images, labels);
        model->save("LBPHFace_model.xml");
    }

    QMessageBox::information(this,"title","model finish");
}

void MainWindow::onRecognitionBtn_FacesRecognitionDemo()
{
    Mat gray;
    const int namenumber = 2;//测试的人脸数量
    const string textname[namenumber] = { "cyt", "xtx" };//做一个储存人脸名字的数组

    Ptr<FaceRecognizer> model;
    if(ui->comboBox_2->currentIndex()==0)
    {
        model = model_E;
    }
    else if(ui->comboBox_2->currentIndex()==1)
    {
        model = model_F;
    }
    else if(ui->comboBox_2->currentIndex()==2)
    {
        model = model_L;
    }

    if(FaceDetect(frame,gray)==true)
    {

        if (!gray.empty())
        {
            cv::equalizeHist(gray, gray);//直方图均衡
            cv::resize(gray, gray, cv::Size(92, 112));//裁剪
            int label ;//= model->predict(gray);
            double confidence;
            model->predict(gray, label, confidence);
            printf("label:%d,confidence:%.4lf\n",label,confidence);
            fflush(NULL);
            cv::putText(frame, textname[label], Point(50, 50), FONT_HERSHEY_DUPLEX, 3, Scalar(230, 255, 0), 2);//model->predict(frame) = predictLabel 名字写在 1 1
            showImage_FacesRecognitionDemo(frame);
            //        imshow("Face Recogniton", frame);
            //        waitKey(-1);
        }
//        SVMpredict(gray);
    }


}

void MainWindow::onTrainSvm()
{
    trainSVM_FacesRecognitionDemo();
}
