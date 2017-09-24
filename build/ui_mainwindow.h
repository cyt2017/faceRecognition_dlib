/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.8.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralWidget;
    QLabel *label_org;
    QLabel *label_gray;
    QPushButton *captureBtn;
    QPushButton *saveBtn;
    QComboBox *comboBox;
    QPushButton *trainBtn;
    QPushButton *recognitionBtn;
    QComboBox *comboBox_2;
    QPushButton *trainSVMBtn;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(805, 587);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        label_org = new QLabel(centralWidget);
        label_org->setObjectName(QStringLiteral("label_org"));
        label_org->setGeometry(QRect(20, 20, 640, 480));
        label_org->setFrameShape(QFrame::Box);
        label_gray = new QLabel(centralWidget);
        label_gray->setObjectName(QStringLiteral("label_gray"));
        label_gray->setGeometry(QRect(680, 20, 92, 112));
        label_gray->setFrameShape(QFrame::Box);
        captureBtn = new QPushButton(centralWidget);
        captureBtn->setObjectName(QStringLiteral("captureBtn"));
        captureBtn->setGeometry(QRect(680, 150, 101, 41));
        saveBtn = new QPushButton(centralWidget);
        saveBtn->setObjectName(QStringLiteral("saveBtn"));
        saveBtn->setGeometry(QRect(730, 210, 51, 31));
        comboBox = new QComboBox(centralWidget);
        comboBox->setObjectName(QStringLiteral("comboBox"));
        comboBox->setGeometry(QRect(680, 210, 41, 31));
        trainBtn = new QPushButton(centralWidget);
        trainBtn->setObjectName(QStringLiteral("trainBtn"));
        trainBtn->setGeometry(QRect(678, 280, 101, 31));
        recognitionBtn = new QPushButton(centralWidget);
        recognitionBtn->setObjectName(QStringLiteral("recognitionBtn"));
        recognitionBtn->setGeometry(QRect(680, 410, 101, 31));
        comboBox_2 = new QComboBox(centralWidget);
        comboBox_2->setObjectName(QStringLiteral("comboBox_2"));
        comboBox_2->setGeometry(QRect(680, 330, 101, 31));
        trainSVMBtn = new QPushButton(centralWidget);
        trainSVMBtn->setObjectName(QStringLiteral("trainSVMBtn"));
        trainSVMBtn->setGeometry(QRect(690, 470, 89, 25));
        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 805, 25));
        MainWindow->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", Q_NULLPTR));
        label_org->setText(QApplication::translate("MainWindow", "TextLabel", Q_NULLPTR));
        label_gray->setText(QApplication::translate("MainWindow", "TextLabel", Q_NULLPTR));
        captureBtn->setText(QApplication::translate("MainWindow", "capture", Q_NULLPTR));
        saveBtn->setText(QApplication::translate("MainWindow", "save", Q_NULLPTR));
        comboBox->clear();
        comboBox->insertItems(0, QStringList()
         << QApplication::translate("MainWindow", "1", Q_NULLPTR)
         << QApplication::translate("MainWindow", "2", Q_NULLPTR)
        );
        trainBtn->setText(QApplication::translate("MainWindow", "train", Q_NULLPTR));
        recognitionBtn->setText(QApplication::translate("MainWindow", "recognition", Q_NULLPTR));
        comboBox_2->clear();
        comboBox_2->insertItems(0, QStringList()
         << QApplication::translate("MainWindow", "eigenFace", Q_NULLPTR)
         << QApplication::translate("MainWindow", "fisherFace", Q_NULLPTR)
         << QApplication::translate("MainWindow", "LBPHFace", Q_NULLPTR)
        );
        trainSVMBtn->setText(QApplication::translate("MainWindow", "train SVM", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
