/*
 * Copyright (c) 2022 HiSilicon (Shanghai) Technologies CO., LIMITED.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <hi_task.h>

#include <hi_wifi_api.h>
#include <hi_mux.h>
#include <hi_io.h>
#include <hi_gpio.h>
#include "iot_config.h"
#include "iot_log.h"
#include "iot_main.h"
#include "iot_profile.h"


#include <stdio.h>
#include <unistd.h>
#include <string.h>

#include "iot_gpio_ex.h"
#include "ohos_init.h"
#include "cmsis_os2.h"
#include "iot_gpio.h"
#include "iot_uart.h"
#include "hi_uart.h"
#include "iot_watchdog.h"
#include "iot_errno.h"



/* attribute initiative to report */
#define TAKE_THE_INITIATIVE_TO_REPORT
#define ONE_SECOND                          (1000)
/* oc request id */
#define CN_COMMADN_INDEX                    "commands/request_id="
#define WECHAT_SUBSCRIBE_LIGHT              "light"
#define WECHAT_SUBSCRIBE_LIGHT_ON_STATE     "1"
#define WECHAT_SUBSCRIBE_LIGHT_OFF_STATE    "0"
#define UART_BUFF_SIZE 2048
#define U_SLEEP_TIME   15000
int g_ligthStatus = -1;
 uint32_t len = 0;
 char *answer =  "ok";

char uartReadBuff[UART_BUFF_SIZE] = "abc";
char uartReadBuff2[UART_BUFF_SIZE] ="abc";
typedef void (*FnMsgCallBack)(hi_gpio_value val);

typedef struct FunctionCallback {
    hi_bool  stop;
    hi_u32 conLost;
    hi_u32 queueID;
    hi_u32 iotTaskID;
    FnMsgCallBack    msgCallBack;
}FunctionCallback;
FunctionCallback g_functinoCallback;
void Uart1GpioInit(void)
{
    IoTGpioInit(IOT_IO_NAME_GPIO_0);
    // 设置GPIO0的管脚复用关系为UART1_TX Set the pin reuse relationship of GPIO0 to UART1_ TX
    IoSetFunc(IOT_IO_NAME_GPIO_0, IOT_IO_FUNC_GPIO_0_UART1_TXD);
    IoTGpioInit(IOT_IO_NAME_GPIO_1);
    // 设置GPIO1的管脚复用关系为UART1_RX Set the pin reuse relationship of GPIO1 to UART1_ RX
    IoSetFunc(IOT_IO_NAME_GPIO_1, IOT_IO_FUNC_GPIO_1_UART1_RXD);
}

void Uart1Config(void)
{
    uint32_t ret;
    /* 初始化UART配置，波特率 9600，数据bit为8,停止位1，奇偶校验为NONE */
    /* Initialize UART configuration, baud rate is 9600, data bit is 8, stop bit is 1, parity is NONE */
    IotUartAttribute uart_attr = {
        .baudRate = 115200,
        .dataBits = 8,
        .stopBits = 1,
        .parity = 0,
    };
    ret = IoTUartInit(HI_UART_IDX_1, &uart_attr);
    if (ret != IOT_SUCCESS) {
        printf("Init Uart1 Falied Error No : %d\n", ret);
        return;
    }
}

char * UartTask(void)
{
   
   char *answer = NULL;
    answer = "ok"; /* 对应手势里面字符串 */
    uint32_t count = 0;
    uint32_t len = 0;
    
    memset(uartReadBuff,0,UART_BUFF_SIZE);
    // 对UART1的一些初始化 Some initialization of UART1
    Uart1GpioInit();
    // 对UART1参数的一些配置 Some configurations of UART1 parameters
    Uart1Config();


       
while(1){
   
       printf("%s\n","Receiving message");
        len = IoTUartRead(HI_UART_IDX_1, uartReadBuff, UART_BUFF_SIZE);
       if (len > 0) {//如果长度大于零，接收成功，此时发信息
            // 把接收到的数据打印出来 Print the received data
            // printf("Uart Read Data is: [ %d ] %s \r\n",len, uartReadBuff);
            printf("%d\n",len );//打印接收的长度
       printf("%s\n",uartReadBuff);
            IoTUartWrite(HI_UART_IDX_1, (unsigned char*)answer, strlen(answer));//发送确认信息
        usleep(U_SLEEP_TIME);
         strcpy(uartReadBuff2,uartReadBuff);
         memset(uartReadBuff,0,UART_BUFF_SIZE);
        

           usleep(15000);
        }//如果没读到就重新去读
       
            
        
}   
        
}
/* CPU Sleep time Set */
unsigned int TaskMsleep(unsigned int ms)
{
    if (ms <= 0) {
        return HI_ERR_FAILURE;
    }
    return hi_sleep((hi_u32)ms);
}

static void DeviceConfigInit(hi_gpio_value val)
{
    hi_io_set_func(HI_IO_NAME_GPIO_9, HI_IO_FUNC_GPIO_9_GPIO);
    hi_gpio_set_dir(HI_GPIO_IDX_9, HI_GPIO_DIR_OUT);
    hi_gpio_set_ouput_val(HI_GPIO_IDX_9, val);
}

static int  DeviceMsgCallback(FnMsgCallBack msgCallBack)
{
    g_functinoCallback.msgCallBack = msgCallBack;
    return 0;
}

static void wechatControlDeviceMsg(hi_gpio_value val)
{
    DeviceConfigInit(val);
}

// < this is the callback function, set to the mqtt, and if any messages come, it will be called
// < The payload here is the json string
static void DemoMsgRcvCallBack(int qos, const char *topic, const char *payload)
{
    IOT_LOG_DEBUG("RCVMSG:QOS:%d TOPIC:%s PAYLOAD:%s\r\n", qos, topic, payload);
    /* 云端下发命令后，板端的操作处理 */
    if (strstr(payload, WECHAT_SUBSCRIBE_LIGHT) != NULL) {
        if (strstr(payload, WECHAT_SUBSCRIBE_LIGHT_OFF_STATE) != NULL) {
            wechatControlDeviceMsg(HI_GPIO_VALUE1);
            g_ligthStatus = HI_FALSE;
        } else {
            wechatControlDeviceMsg(HI_GPIO_VALUE0);
            g_ligthStatus = HI_TRUE;
        }
    }
    return HI_NULL;
}

/* publish sample */
hi_void IotPublishSample(void)
{
    memset(uartReadBuff,0,UART_BUFF_SIZE);
    int i=0;
    while(1){

       printf("%s\n","Receving message");
       usleep(1000);
        len = IoTUartRead(HI_UART_IDX_1, uartReadBuff, UART_BUFF_SIZE);
       if (len > 0) {//如果长度大于零，接收成功，此时发信息
            // 把接收到的数据打印出来 Print the received data
            // printf("Uart Read Data is: [ %d ] %s \r\n",len, uartReadBuff);
            printf("%d\n",len );//打印接收的长度
       printf("%s\n",uartReadBuff);
            IoTUartWrite(HI_UART_IDX_1, (unsigned char*)answer, strlen(answer));//发送确认信息
        usleep(U_SLEEP_TIME);
        break;
        //  strcpy(uartReadBuff2,uartReadBuff);
        //  memset(uartReadBuff,0,UART_BUFF_SIZE);
      
        }//如果没读到就重新去读
        i++;
        if(i>10){
           break; 
        }        
}
printf("uartdata:",uartReadBuff);
if(len!=0){
    strcpy(uartReadBuff2,uartReadBuff);
}
    /* reported attribute */
    WeChatProfile weChatProfile = {
        .subscribeType = "type",
        .status.subState = "state",
        .status.subReport = "reported",
        .status.reportVersion = "version",
        .status.Token = "clientToken",
        /* report motor */
        // .reportAction.subDeviceActionMotor = "motor",
        // .reportAction.motorActionStatus = 0, /* 0 : motor off */
        /* report temperature */
        .reportAction.subDeviceActionTemperature = "data",
        .reportAction.temperatureData = (const char * const)uartReadBuff2 , /* 30 :temperature data */
        /* report humidity */
        // .reportAction.subDeviceActionHumidity = "humidity",
        // .reportAction.humidityActionData = 100, /* humidity data */
        // /* report light_intensity */
        // .reportAction.subDeviceActionLightIntensity = "light_intensity",
        // .reportAction.lightIntensityActionData = 60, /* 60 : light_intensity */
    };

    /* report light */
    if (g_ligthStatus == HI_TRUE) {
        weChatProfile.reportAction.subDeviceActionLight = "light";
        weChatProfile.reportAction.lightActionStatus = 1; /* 1: light on */
    } else if (g_ligthStatus == HI_FALSE) {
        weChatProfile.reportAction.subDeviceActionLight = "light";
        weChatProfile.reportAction.lightActionStatus = 0; /* 0: light off */
    } else {
        weChatProfile.reportAction.subDeviceActionLight = "light";
        weChatProfile.reportAction.lightActionStatus = 0; /* 0: light off */
    }
    /* profile report */
    IoTProfilePropertyReport(CONFIG_USER_ID, &weChatProfile);
}

// < this is the demo main task entry,here we will set the wifi/cjson/mqtt ready and
// < wait if any work to do in the while
static hi_void *DemoEntry(const char *arg)
{
    WifiStaReadyWait();
    cJsonInit();
    IoTMain();
    /* 云端下发回调 */
    IoTSetMsgCallback(DemoMsgRcvCallBack);
    /* 主动上报 */
    // 对UART1的一些初始化 Some initialization of UART1
    Uart1GpioInit();
    // 对UART1参数的一些配置 Some configurations of UART1 parameters
    Uart1Config();
#ifdef TAKE_THE_INITIATIVE_TO_REPORT
    while (1) {
        /* 用户可以在这调用发布函数进行发布，需要用户自己写调用函数 */
        IotPublishSample(); // 发布例程
#endif
        TaskMsleep(ONE_SECOND);
    }
    return NULL;
}

// < This is the demo entry, we create a task here,
// and all the works has been done in the demo_entry
#define CN_IOT_TASK_STACKSIZE  0x1000
#define CN_IOT_TASK_PRIOR 25
#define CN_IOT_TASK_NAME "IOTDEMO"

static void AppDemoIot(void)
{
    osThreadAttr_t attr;
    osThreadAttr_t attr2;
    IoTWatchDogDisable();

    attr.name = "IOTDEMO";
    attr.attr_bits = 0U;
    attr.cb_mem = NULL;
    attr.cb_size = 0U;
    attr.stack_mem = NULL;
    attr.stack_size = CN_IOT_TASK_STACKSIZE;
    attr.priority = CN_IOT_TASK_PRIOR;
    attr2.name = "UartTask";
    attr2.attr_bits = 0U;
    attr2.cb_mem = NULL;
    attr2.cb_size = 0U;
    attr2.stack_mem = NULL;
    attr2.stack_size = 3 * 1024; // 任务栈大小*1024 stack size 5*1024
    attr2.priority = osPriorityNormal;

    if (osThreadNew((osThreadFunc_t)DemoEntry, NULL, &attr) == NULL) {
        printf("[mqtt] Falied to create IOTDEMO!\n");
    }
    
    // if (osThreadNew((osThreadFunc_t)UartTask, NULL, &attr2) == NULL) {
    //     printf("[UartTask] Failed to create UartTask!\n");
    // }
}

SYS_RUN(AppDemoIot);