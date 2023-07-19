""" ━━━━━━神兽出没━━━━━━ 
 　　　┏┓　　　┏┓ 
 　　┃　　　　　　　┃ 
 　　┃　　　━　　　┃ 
 　　┃　┳┛　┗┳　┃ 
 　　┃　　　　　　　┃ 
 　　┃　　　┻　　　┃ 
 　　┃　　　　　　　┃ 
 　　┗━┓　　　┏━┛Code is far away from bug with the animal rotecting 
 　　　　┃　　　┃ 神兽保佑,代码无bug 
 　　　　┃　　　┃ 
 　　　　┃　　　┗━━━┓ 
 　　　　┃　　　　　　　┣┓ 
 　　　　┃　　　　　　　┏┛ 
 　　  　┗┓┓┏━┳┓┏┛ 
　　　　　┃┫┫　┃┫┫ 
　　　　　┗┻┛　┗┻┛ 
"""
import sys

import qdarkstyle
from PyQt5.QtWidgets import QApplication, QStyleFactory



# try:
#     import smart_classroom_rc
# except ImportError:
#     pass
import torch
from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication
from ui.MainWindow_smart_room import  Ui_smart_classroom
from function_app.MainWindow_smart_room import Ui_smart_classroom as SmartClassroom
from function_app.face_register_app import Face_RegisterApp
from function_app.dynamic_attendance_app  import DynamicAttendanceApp

from function_app.class_concentration_app import ClassConcentrationApp
import socket
torch._C._jit_set_profiling_mode(False)
torch.jit.optimized_execution(False)
# JIT编译器是PyTorch中的一种优化工具，可以将Python代码转换为C++代码以加速模型执行。
# 这些代码可以手工调用，以控制JIT编译器的模式和优化行为。
# torch._C._jit_set_profiling_mode(False)用于关闭JIT编译器的分析模式，该模式用于
# 收集模型执行期间的性能数据，以便进行优化。在分析模式下，JIT编译器会对模型执行进行详细记
# 录并生成性能分析报告。关闭此模式可以减少JIT编译器的内存使用和计算开销，从而提高模型的执行速度。
# torch.jit.optimized_execution(False)用于关闭JIT编译器的优化模式，该模式用于对模型执
# 行进行优化，以提高模型的执行速度。在优化模式下，JIT编译器会对模型执行进行各种优化，如常量
# 折叠、循环拆分和内存优化等。关闭此模式可以降低JIT编译器的计算开销和内存使用，但可能会降低模型的执行速度。
# 值得注意的是，这些代码应该谨慎使用，因为它们可能会影响模型的性能和准确性。除非有特定的需求，
# 否则不建议在生产环境中使用这些代码。

class SmartClassroomApp(QMainWindow, Ui_smart_classroom):

    def __init__(self, parent=None):
        super(SmartClassroomApp, self).__init__(parent)
        self.setupUi(self)


        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # 绑定socket到一个可用的端口上
        self.server_address = ('192.168.200.1', 8888)  # 设置PC端的IP地址和端口号
        self.server_socket.bind(self.server_address)

        # 监听连接
        self.server_socket.listen(5)
        #第二个socket
        self.server_socket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # 绑定socket到一个可用的端口上
        self.server_address2 = ('192.168.200.1', 8887)  # 设置PC端的IP地址和端口号
        self.server_socket2.bind(self.server_address2)

        # 监听连接
        self.server_socket2.listen(1)
        # self.class_widget = In_ClassApp()
        # self.class_widget.setObjectName("in_class_widget")
        # self.tabWidget.addTab(self.class_widget, "课堂监测")
        #
        # self.exam_widget =In_ExamApp()
        # self.exam_widget.setObjectName("in_exam_widget")
        # self.tabWidget.addTab(self.exam_widget, "考试监测")
        #
        self.face_register_widget = Face_RegisterApp(self.server_socket,self.server_socket2)
        self.face_register_widget.setObjectName("face_register_widget")
        self.tabWidget.addTab(self.face_register_widget, "人脸注册")

        self.sign_widget =DynamicAttendanceApp(self.server_socket,self.server_socket2)
        self.sign_widget.setObjectName("sign_widget")
        self.tabWidget.addTab(self.sign_widget, "动态签到")

   

        self.class_concentration_widget = ClassConcentrationApp(self.server_socket,self.server_socket2)
        self.class_concentration_widget.setObjectName("class_concentration_widget")
        self.tabWidget.addTab(self.class_concentration_widget, "课堂专注度分析")
        self.current_tab_widget = self.tabWidget.currentWidget()

        self.current_tab_widget.open()
        def current_tab_change(idx, self=self):
            if self.current_tab_widget is not None:
                self.current_tab_widget.close()
            self.current_tab_widget = self.tabWidget.widget(idx)
            self.current_tab_widget.open()
            print("打开")

        self.tabWidget.currentChanged.connect(current_tab_change)
        # def change_tab_widget(index):
        #     self.tabWidget.widget(index).close()
        #
        # self.tabWidget.currentChanged.connect()
        # _translate = QtCore.QCoreApplication.translate
        # self.tabWidget.setTabText(self.tabWidget.indexOf(self.cheating_detection_widget),
        #                           _translate("MainWindow", "作弊检测"))

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.cheating_detection_widget.close()
        self.face_register_widget.close()
        self.dynamic_attendance_widget.close()
        self.class_concentration_widget.close()
        super(SmartClassroomApp, self).closeEvent(a0)


if __name__ == '__main__':

    app = QApplication(sys.argv)
    # from QcureUi import cure
    #
    # window = cure.Windows(SmartClassroomApp(), 'trayname', True, title='智慧教室')

    window = SmartClassroomApp()
    available_styles = QStyleFactory.keys()
    print(available_styles)
    app.setStyle('windowsvista')


    # run
    window.show()
    sys.exit(app.exec_())
