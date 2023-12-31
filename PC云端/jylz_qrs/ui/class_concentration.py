# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'class_concentration.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ClassConcentration(object):
    def setupUi(self, ClassConcentration):
        ClassConcentration.setObjectName("ClassConcentration")
        ClassConcentration.resize(1240, 769)
        self.verticalLayout_11 = QtWidgets.QVBoxLayout(ClassConcentration)
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_10 = QtWidgets.QLabel(ClassConcentration)
        self.label_10.setObjectName("label_10")
        self.horizontalLayout_8.addWidget(self.label_10)
        self.video_source_txt = QtWidgets.QLineEdit(ClassConcentration)
        self.video_source_txt.setObjectName("video_source_txt")
        self.horizontalLayout_8.addWidget(self.video_source_txt)
        self.open_source_btn = QtWidgets.QPushButton(ClassConcentration)
        self.open_source_btn.setObjectName("open_source_btn")
        self.horizontalLayout_8.addWidget(self.open_source_btn)
        self.close_source_btn = QtWidgets.QPushButton(ClassConcentration)
        self.close_source_btn.setObjectName("close_source_btn")
        self.horizontalLayout_8.addWidget(self.close_source_btn)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem)
        self.verticalLayout_11.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout()
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.line_7 = QtWidgets.QFrame(ClassConcentration)
        self.line_7.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.verticalLayout_7.addWidget(self.line_7)
        self.label_11 = QtWidgets.QLabel(ClassConcentration)
        self.label_11.setObjectName("label_11")
        self.verticalLayout_7.addWidget(self.label_11)
        self.video_resource_list = QtWidgets.QListWidget(ClassConcentration)
        self.video_resource_list.setObjectName("video_resource_list")
        self.verticalLayout_7.addWidget(self.video_resource_list)
        self.line_8 = QtWidgets.QFrame(ClassConcentration)
        self.line_8.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_8.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_8.setObjectName("line_8")
        self.verticalLayout_7.addWidget(self.line_8)
        self.label_12 = QtWidgets.QLabel(ClassConcentration)
        self.label_12.setObjectName("label_12")
        self.verticalLayout_7.addWidget(self.label_12)
        self.video_resource_file_list = QtWidgets.QListWidget(ClassConcentration)
        self.video_resource_file_list.setObjectName("video_resource_file_list")
        self.verticalLayout_7.addWidget(self.video_resource_file_list)
        self.verticalLayout_2.addLayout(self.verticalLayout_7)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.real_time_catch_lbl_2 = QtWidgets.QLabel(ClassConcentration)
        self.real_time_catch_lbl_2.setStyleSheet("font: 12pt \"华文琥珀\";")
        self.real_time_catch_lbl_2.setObjectName("real_time_catch_lbl_2")
        self.verticalLayout.addWidget(self.real_time_catch_lbl_2)
        self.primary_factor_img = QtWidgets.QLabel(ClassConcentration)
        self.primary_factor_img.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.primary_factor_img.sizePolicy().hasHeightForWidth())
        self.primary_factor_img.setSizePolicy(sizePolicy)
        self.primary_factor_img.setMinimumSize(QtCore.QSize(0, 200))
        self.primary_factor_img.setSizeIncrement(QtCore.QSize(0, 0))
        self.primary_factor_img.setAutoFillBackground(False)
        self.primary_factor_img.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.primary_factor_img.setText("")
        self.primary_factor_img.setObjectName("primary_factor_img")
        self.verticalLayout.addWidget(self.primary_factor_img)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        self.verticalLayout_2.setStretch(0, 6)
        self.verticalLayout_2.setStretch(1, 4)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.line = QtWidgets.QFrame(ClassConcentration)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout_2.addWidget(self.line)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout()
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.label_13 = QtWidgets.QLabel(ClassConcentration)
        self.label_13.setStyleSheet("font: 12pt \"华文琥珀\";")
        self.label_13.setObjectName("label_13")
        self.horizontalLayout_10.addWidget(self.label_13)
        self.show_box_ckb = QtWidgets.QCheckBox(ClassConcentration)
        self.show_box_ckb.setChecked(True)
        self.show_box_ckb.setObjectName("show_box_ckb")
        self.horizontalLayout_10.addWidget(self.show_box_ckb)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_10.addItem(spacerItem1)
        self.verticalLayout_9.addLayout(self.horizontalLayout_10)
        self.video_screen = QtWidgets.QLabel(ClassConcentration)
        self.video_screen.setAutoFillBackground(False)
        self.video_screen.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.video_screen.setFrameShadow(QtWidgets.QFrame.Plain)
        self.video_screen.setText("")
        self.video_screen.setObjectName("video_screen")
        self.verticalLayout_9.addWidget(self.video_screen)
        self.video_process_bar = QtWidgets.QSlider(ClassConcentration)
        self.video_process_bar.setAutoFillBackground(False)
        self.video_process_bar.setStyleSheet(" QSlider {\n"
"    background-color: rgba(22, 22, 22, 0.7);\n"
"    border-radius: 5px;\n"
"}\n"
" \n"
"QSlider::sub-page:horizontal {\n"
"    background-color: #FF7826;\n"
"    height:4px;\n"
"    border-radius: 2px;\n"
"}\n"
" \n"
"QSlider::add-page:horizontal {\n"
"    background-color: #7A7B79;\n"
"    height:4px;\n"
"    border-radius: 2px;\n"
"}\n"
" \n"
"QSlider::groove:horizontal {\n"
"    background:transparent;\n"
"    height:10px;\n"
"}\n"
" \n"
"QSlider::handle:horizontal {\n"
"    height: 10px;\n"
"    width: 10px;\n"
"    margin: 0px -2px 0px -2px;\n"
"    border-radius: 5px;\n"
"    background: white;\n"
"}")
        self.video_process_bar.setMinimum(-1)
        self.video_process_bar.setMaximum(-1)
        self.video_process_bar.setProperty("value", -1)
        self.video_process_bar.setOrientation(QtCore.Qt.Horizontal)
        self.video_process_bar.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.video_process_bar.setTickInterval(0)
        self.video_process_bar.setObjectName("video_process_bar")
        self.verticalLayout_9.addWidget(self.video_process_bar)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.time_process_label = QtWidgets.QLabel(ClassConcentration)
        self.time_process_label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.time_process_label.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.time_process_label.setObjectName("time_process_label")
        self.horizontalLayout_11.addWidget(self.time_process_label)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_11.addItem(spacerItem2)
        self.play_video_btn = QtWidgets.QPushButton(ClassConcentration)
        self.play_video_btn.setObjectName("play_video_btn")
        self.horizontalLayout_11.addWidget(self.play_video_btn)
        self.stop_playing_btn = QtWidgets.QPushButton(ClassConcentration)
        self.stop_playing_btn.setObjectName("stop_playing_btn")
        self.horizontalLayout_11.addWidget(self.stop_playing_btn)
        self.horizontalLayout_11.setStretch(0, 1)
        self.horizontalLayout_11.setStretch(1, 3)
        self.horizontalLayout_11.setStretch(2, 1)
        self.horizontalLayout_11.setStretch(3, 1)
        self.verticalLayout_9.addLayout(self.horizontalLayout_11)
        self.verticalLayout_9.setStretch(1, 8)
        self.verticalLayout_9.setStretch(3, 1)
        self.verticalLayout_3.addLayout(self.verticalLayout_9)
        self.line_10 = QtWidgets.QFrame(ClassConcentration)
        self.line_10.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_10.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_10.setObjectName("line_10")
        self.verticalLayout_3.addWidget(self.line_10)
        self.verticalLayout_10 = QtWidgets.QVBoxLayout()
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.real_time_catch_ico = QtWidgets.QLabel(ClassConcentration)
        self.real_time_catch_ico.setMaximumSize(QtCore.QSize(25, 25))
        self.real_time_catch_ico.setScaledContents(True)
        self.real_time_catch_ico.setObjectName("real_time_catch_ico")
        self.horizontalLayout_12.addWidget(self.real_time_catch_ico)
        self.real_time_catch_lbl = QtWidgets.QLabel(ClassConcentration)
        self.real_time_catch_lbl.setStyleSheet("font: 12pt \"华文琥珀\";")
        self.real_time_catch_lbl.setObjectName("real_time_catch_lbl")
        self.horizontalLayout_12.addWidget(self.real_time_catch_lbl)
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.label_14 = QtWidgets.QLabel(ClassConcentration)
        self.label_14.setObjectName("label_14")
        self.horizontalLayout_13.addWidget(self.label_14)
        self.line_data_limit_spin = QtWidgets.QSpinBox(ClassConcentration)
        self.line_data_limit_spin.setMinimum(20)
        self.line_data_limit_spin.setMaximum(5000)
        self.line_data_limit_spin.setProperty("value", 60)
        self.line_data_limit_spin.setObjectName("line_data_limit_spin")
        self.horizontalLayout_13.addWidget(self.line_data_limit_spin)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_13.addItem(spacerItem3)
        self.horizontalLayout_12.addLayout(self.horizontalLayout_13)
        self.verticalLayout_10.addLayout(self.horizontalLayout_12)
        self.primary_level_img = QtWidgets.QLabel(ClassConcentration)
        self.primary_level_img.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.primary_level_img.sizePolicy().hasHeightForWidth())
        self.primary_level_img.setSizePolicy(sizePolicy)
        self.primary_level_img.setMinimumSize(QtCore.QSize(0, 200))
        self.primary_level_img.setSizeIncrement(QtCore.QSize(0, 0))
        self.primary_level_img.setAutoFillBackground(False)
        self.primary_level_img.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.primary_level_img.setText("")
        self.primary_level_img.setObjectName("primary_level_img")
        self.verticalLayout_10.addWidget(self.primary_level_img)
        self.verticalLayout_10.setStretch(1, 1)
        self.verticalLayout_3.addLayout(self.verticalLayout_10)
        self.verticalLayout_3.setStretch(0, 1)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        self.verticalLayout_8 = QtWidgets.QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.real_time_catch_lbl_3 = QtWidgets.QLabel(ClassConcentration)
        self.real_time_catch_lbl_3.setStyleSheet("font: 12pt \"华文琥珀\";")
        self.real_time_catch_lbl_3.setObjectName("real_time_catch_lbl_3")
        self.verticalLayout_5.addWidget(self.real_time_catch_lbl_3)
        self.action_level_img = QtWidgets.QLabel(ClassConcentration)
        self.action_level_img.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.action_level_img.sizePolicy().hasHeightForWidth())
        self.action_level_img.setSizePolicy(sizePolicy)
        self.action_level_img.setMinimumSize(QtCore.QSize(0, 200))
        self.action_level_img.setSizeIncrement(QtCore.QSize(0, 0))
        self.action_level_img.setAutoFillBackground(False)
        self.action_level_img.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.action_level_img.setText("")
        self.action_level_img.setObjectName("action_level_img")
        self.verticalLayout_5.addWidget(self.action_level_img)
        self.verticalLayout_8.addLayout(self.verticalLayout_5)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.real_time_catch_lbl_4 = QtWidgets.QLabel(ClassConcentration)
        self.real_time_catch_lbl_4.setStyleSheet("font: 12pt \"华文琥珀\";")
        self.real_time_catch_lbl_4.setObjectName("real_time_catch_lbl_4")
        self.verticalLayout_4.addWidget(self.real_time_catch_lbl_4)
        self.face_level_img = QtWidgets.QLabel(ClassConcentration)
        self.face_level_img.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.face_level_img.sizePolicy().hasHeightForWidth())
        self.face_level_img.setSizePolicy(sizePolicy)
        self.face_level_img.setMinimumSize(QtCore.QSize(0, 200))
        self.face_level_img.setSizeIncrement(QtCore.QSize(0, 0))
        self.face_level_img.setAutoFillBackground(False)
        self.face_level_img.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.face_level_img.setText("")
        self.face_level_img.setObjectName("face_level_img")
        self.verticalLayout_4.addWidget(self.face_level_img)
        self.verticalLayout_8.addLayout(self.verticalLayout_4)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.real_time_catch_lbl_5 = QtWidgets.QLabel(ClassConcentration)
        self.real_time_catch_lbl_5.setStyleSheet("font: 12pt \"华文琥珀\";")
        self.real_time_catch_lbl_5.setObjectName("real_time_catch_lbl_5")
        self.verticalLayout_6.addWidget(self.real_time_catch_lbl_5)
        self.head_pose_level_img = QtWidgets.QLabel(ClassConcentration)
        self.head_pose_level_img.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.head_pose_level_img.sizePolicy().hasHeightForWidth())
        self.head_pose_level_img.setSizePolicy(sizePolicy)
        self.head_pose_level_img.setMinimumSize(QtCore.QSize(0, 200))
        self.head_pose_level_img.setSizeIncrement(QtCore.QSize(0, 0))
        self.head_pose_level_img.setAutoFillBackground(False)
        self.head_pose_level_img.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.head_pose_level_img.setText("")
        self.head_pose_level_img.setObjectName("head_pose_level_img")
        self.verticalLayout_6.addWidget(self.head_pose_level_img)
        self.verticalLayout_8.addLayout(self.verticalLayout_6)
        self.horizontalLayout.addLayout(self.verticalLayout_8)
        self.horizontalLayout.setStretch(0, 6)
        self.horizontalLayout.setStretch(1, 4)
        self.horizontalLayout_2.addLayout(self.horizontalLayout)
        self.horizontalLayout_2.setStretch(0, 4)
        self.horizontalLayout_2.setStretch(2, 10)
        self.verticalLayout_11.addLayout(self.horizontalLayout_2)

        self.retranslateUi(ClassConcentration)
        QtCore.QMetaObject.connectSlotsByName(ClassConcentration)

    def retranslateUi(self, ClassConcentration):
        _translate = QtCore.QCoreApplication.translate
        ClassConcentration.setWindowTitle(_translate("ClassConcentration", "专注度分析"))
        self.label_10.setText(_translate("ClassConcentration", "视频源"))
        self.video_source_txt.setText(_translate("ClassConcentration", "resource/videos/front_cheat.mp4"))
        self.open_source_btn.setText(_translate("ClassConcentration", "开启源"))
        self.close_source_btn.setText(_translate("ClassConcentration", "关闭源"))
        self.label_11.setText(_translate("ClassConcentration", "视频通道"))
        self.label_12.setText(_translate("ClassConcentration", "本地视频"))
        self.real_time_catch_lbl_2.setText(_translate("ClassConcentration", "评价因素权重（一致性）"))
        self.label_13.setText(_translate("ClassConcentration", "课堂专注度"))
        self.show_box_ckb.setText(_translate("ClassConcentration", "显示边框"))
        self.time_process_label.setText(_translate("ClassConcentration", "00:00:00/00:00:00"))
        self.play_video_btn.setText(_translate("ClassConcentration", "播放"))
        self.stop_playing_btn.setText(_translate("ClassConcentration", "暂停"))
        self.real_time_catch_ico.setText(_translate("ClassConcentration", "lbl"))
        self.real_time_catch_lbl.setText(_translate("ClassConcentration", "群体专注度曲线"))
        self.label_14.setText(_translate("ClassConcentration", "上限"))
        self.real_time_catch_lbl_3.setText(_translate("ClassConcentration", "行为专注度评分曲线"))
        self.real_time_catch_lbl_4.setText(_translate("ClassConcentration", "情绪专注度评分曲线"))
        self.real_time_catch_lbl_5.setText(_translate("ClassConcentration", "头部姿态专注度评分曲线"))
