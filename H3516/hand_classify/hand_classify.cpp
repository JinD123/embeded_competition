/*
 * Copyright (c) 2022 HiSilicon (Shanghai) Technologies CO., LIMITED.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * 该文件提供了基于yolov2的手部检测以及基于resnet18的手势识别，属于两个wk串行推理。
 * 该文件提供了手部检测和手势识别的模型加载、模型卸载、模型推理以及AI flag业务处理的API接口。
 * 若一帧图像中出现多个手，我们通过算法将最大手作为目标手送分类网进行推理，
 * 并将目标手标记为绿色，其他手标记为红色。
 *
 * This file provides hand detection based on yolov2 and gesture recognition based on resnet18,
 * which belongs to two wk serial inferences. This file provides API interfaces for model loading,
 * model unloading, model reasoning, and AI flag business processing for hand detection
 * and gesture recognition. If there are multiple hands in one frame of image,
 * we use the algorithm to use the largest hand as the target hand for inference,
 * and mark the target hand as green and the other hands as red.
 */
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include<pthread.h>
#include "sample_comm_nnie.h"
#include "sample_media_ai.h"
#include "ai_infer_process.h"
#include "yolov2_hand_detect.h"
#include "vgs_img.h"
#include "ive_img.h"
#include "misc_util.h"
#include "hisignalling.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <iostream>
#include <opencv2/core.hpp>


#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "hi_debug.h"
#include "hi_comm_ive.h"
#include "smp_color_space_convert.h"
#include "sample_comm_nnie.h"
#include "sample_comm_ive.h"
#include "sample_media_ai.h"
#include "vgs_img.h"
#include "misc_util.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <poll.h>
#include "hi_debug.h"
#include "hi_comm_ive.h"
#include "smp_color_space_convert.h"
#include "sample_comm_nnie.h"
#include "sample_comm_ive.h"
#include "sample_media_ai.h"
#include "vgs_img.h"
#include "misc_util.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include "hi_type.h"
#define HAND_FRM_WIDTH     640
#define HAND_FRM_HEIGHT    480
#define DETECT_OBJ_MAX     32
#define RET_NUM_MAX        4
#define DRAW_RETC_THICK    4   // Draw the width of the line
#define WIDTH_LIMIT        32
#define HEIGHT_LIMIT       32
#define IMAGE_WIDTH        224  // The resolution of the model IMAGE sent to the classification is 224*224
#define IMAGE_HEIGHT       224
#define MODEL_FILE_GESTURE    "/userdata/models/hand_classify/hand_gesture.wk" // darknet framework wk model
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
using namespace cv;
#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif /* End of #ifdef __cplusplus */
static IVE_SRC_IMAGE_S pstSrc;
static IVE_DST_IMAGE_S pstDst;
static IVE_CSC_CTRL_S stCscCtrl;
static int biggestBoxIndex;
static IVE_IMAGE_S img;
static DetectObjInfo objs[DETECT_OBJ_MAX] = {0};
static RectBox boxs[DETECT_OBJ_MAX] = {0};
static RectBox objBoxs[DETECT_OBJ_MAX] = {0};
static RectBox remainingBoxs[DETECT_OBJ_MAX] = {0};
static RectBox cnnBoxs[DETECT_OBJ_MAX] = {0}; // Store the results of the classification network
static RecogNumInfo numInfo[RET_NUM_MAX] = {0};
static IVE_IMAGE_S imgIn;
static IVE_IMAGE_S imgDst;
static VIDEO_FRAME_INFO_S frmIn;
static VIDEO_FRAME_INFO_S frmDst;

int client_socket ;
int i=0;
/*
 * 加载手部检测和手势分类模型
 * Load hand detect and classify model
 */




static HI_VOID IveImageParamCfg(IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
    VIDEO_FRAME_INFO_S *srcFrame)
{
    pstSrc->enType = IVE_IMAGE_TYPE_YUV420SP;
    pstSrc->au64VirAddr[0] = srcFrame->stVFrame.u64VirAddr[0];
    pstSrc->au64VirAddr[1] = srcFrame->stVFrame.u64VirAddr[1];
    pstSrc->au64VirAddr[2] = srcFrame->stVFrame.u64VirAddr[2]; // 2: Image data virtual address

    pstSrc->au64PhyAddr[0] = srcFrame->stVFrame.u64PhyAddr[0];
    pstSrc->au64PhyAddr[1] = srcFrame->stVFrame.u64PhyAddr[1];
    pstSrc->au64PhyAddr[2] = srcFrame->stVFrame.u64PhyAddr[2]; // 2: Image data physical address

    pstSrc->au32Stride[0] = srcFrame->stVFrame.u32Stride[0];
    pstSrc->au32Stride[1] = srcFrame->stVFrame.u32Stride[1];
    pstSrc->au32Stride[2] = srcFrame->stVFrame.u32Stride[2]; // 2: Image data span

    pstSrc->u32Width = srcFrame->stVFrame.u32Width;
    pstSrc->u32Height = srcFrame->stVFrame.u32Height;

    pstDst->enType = IVE_IMAGE_TYPE_U8C3_PACKAGE;
    pstDst->u32Width = pstSrc->u32Width;
    pstDst->u32Height = pstSrc->u32Height;
    pstDst->au32Stride[0] = pstSrc->au32Stride[0];
    pstDst->au32Stride[1] = 0;
    pstDst->au32Stride[2] = 0; // 2: Image data span
}
static HI_S32 yuvFrame2rgb(VIDEO_FRAME_INFO_S *srcFrame, IPC_IMAGE *dstImage)
{
    IVE_HANDLE hIveHandle;
    HI_S32 s32Ret = 0;
    stCscCtrl.enMode = IVE_CSC_MODE_PIC_BT709_YUV2RGB; // IVE_CSC_MODE_VIDEO_BT601_YUV2RGB
    IveImageParamCfg(&pstSrc, &pstDst, srcFrame);

    s32Ret = HI_MPI_SYS_MmzAlloc_Cached(&pstDst.au64PhyAddr[0], (void **)&pstDst.au64VirAddr[0],
        "User", HI_NULL, pstDst.u32Height*pstDst.au32Stride[0] * 3); // 3: multiple
    if (HI_SUCCESS != s32Ret) {
        HI_MPI_SYS_MmzFree(pstDst.au64PhyAddr[0], (void *)pstDst.au64VirAddr[0]);
        SAMPLE_PRT("HI_MPI_SYS_MmzFree err\n");
        return s32Ret;
    }

    s32Ret = HI_MPI_SYS_MmzFlushCache(pstDst.au64PhyAddr[0], (void *)pstDst.au64VirAddr[0],
        pstDst.u32Height*pstDst.au32Stride[0] * 3); // 3: multiple
    if (HI_SUCCESS != s32Ret) {
        HI_MPI_SYS_MmzFree(pstDst.au64PhyAddr[0], (void *)pstDst.au64VirAddr[0]);
        return s32Ret;
    }
    // 3: multiple
    memset_s((void *)pstDst.au64VirAddr[0], pstDst.u32Height*pstDst.au32Stride[0] * 3,
        0, pstDst.u32Height*pstDst.au32Stride[0] * 3); // 3: multiple
    HI_BOOL bInstant = HI_TRUE;

    s32Ret = HI_MPI_IVE_CSC(&hIveHandle, &pstSrc, &pstDst, &stCscCtrl, bInstant);
    if (HI_SUCCESS != s32Ret) {
        HI_MPI_SYS_MmzFree(pstDst.au64PhyAddr[0], (void *)pstDst.au64VirAddr[0]);
        return s32Ret;
    }

    if (HI_TRUE == bInstant) {
        HI_BOOL bFinish = HI_TRUE;
        HI_BOOL bBlock = HI_TRUE;
        s32Ret = HI_MPI_IVE_Query(hIveHandle, &bFinish, bBlock);
        while (HI_ERR_IVE_QUERY_TIMEOUT == s32Ret) {
            usleep(100); // 100: usleep time
            s32Ret = HI_MPI_IVE_Query(hIveHandle, &bFinish, bBlock);
        }
    }
    dstImage->u64PhyAddr = pstDst.au64PhyAddr[0];
    dstImage->u64VirAddr = pstDst.au64VirAddr[0];
    dstImage->u32Width = pstDst.u32Width;
    dstImage->u32Height = pstDst.u32Height;

    return HI_SUCCESS;
}

static HI_S32 frame2Mat(VIDEO_FRAME_INFO_S *srcFrame, Mat &dstMat)
{
    HI_U32 w = srcFrame->stVFrame.u32Width;
    HI_U32 h = srcFrame->stVFrame.u32Height;
    int bufLen = w * h * 3;
    HI_U8 *srcRGB = NULL;
    IPC_IMAGE dstImage;
    if (yuvFrame2rgb(srcFrame, &dstImage) != HI_SUCCESS) {
        SAMPLE_PRT("yuvFrame2rgb err\n");
        return HI_FAILURE;
    }
    srcRGB = (HI_U8 *)dstImage.u64VirAddr;
    dstMat.create(h, w, CV_8UC3);
    memcpy_s(dstMat.data, bufLen * sizeof(HI_U8), srcRGB, bufLen * sizeof(HI_U8));
    HI_MPI_SYS_MmzFree(dstImage.u64PhyAddr, (void *)&(dstImage.u64VirAddr));
    return HI_SUCCESS;
}
HI_S32 Yolo2HandDetectResnetClassifyLoad(uintptr_t* model)
{
    SAMPLE_SVP_NNIE_CFG_S *self = NULL;
    HI_S32 ret;

    ret = CnnCreate(&self, MODEL_FILE_GESTURE);
    *model = ret < 0 ? 0 : (uintptr_t)self;
    HandDetectInit(); // Initialize the hand detection model
    SAMPLE_PRT("Load hand detect claasify model success\n");
    /*
     * Uart串口初始化
     * Uart open init
     */

    
    return ret;
}

/*
 * 卸载手部检测和手势分类模型
 * Unload hand detect and classify model
 */
HI_S32 Yolo2HandDetectResnetClassifyUnload(uintptr_t model)
{
    CnnDestroy((SAMPLE_SVP_NNIE_CFG_S*)model);
    HandDetectExit(); // Uninitialize the hand detection model
    
    SAMPLE_PRT("Unload hand detect claasify model success\n");

    return 0;
}

/*
 * 获得最大的手
 * Get the maximum hand
 */
static HI_S32 GetBiggestHandIndex(RectBox boxs[], int detectNum)
{
    HI_S32 handIndex = 0;
    HI_S32 biggestBoxIndex = handIndex;
    HI_S32 biggestBoxWidth = boxs[handIndex].xmax - boxs[handIndex].xmin + 1;
    HI_S32 biggestBoxHeight = boxs[handIndex].ymax - boxs[handIndex].ymin + 1;
    HI_S32 biggestBoxArea = biggestBoxWidth * biggestBoxHeight;

    for (handIndex = 1; handIndex < detectNum; handIndex++) {
        HI_S32 boxWidth = boxs[handIndex].xmax - boxs[handIndex].xmin + 1;
        HI_S32 boxHeight = boxs[handIndex].ymax - boxs[handIndex].ymin + 1;
        HI_S32 boxArea = boxWidth * boxHeight;
        if (biggestBoxArea < boxArea) {
            biggestBoxArea = boxArea;
            biggestBoxIndex = handIndex;
        }
        biggestBoxWidth = boxs[biggestBoxIndex].xmax - boxs[biggestBoxIndex].xmin + 1;
        biggestBoxHeight = boxs[biggestBoxIndex].ymax - boxs[biggestBoxIndex].ymin + 1;
    }

    if ((biggestBoxWidth == 1) || (biggestBoxHeight == 1) || (detectNum == 0)) {
        biggestBoxIndex = -1;
    }

    return biggestBoxIndex;
}

/*
 * 手势识别信息
 * Hand gesture recognition info
 */
static void HandDetectFlag(const RecogNumInfo resBuf)
{
    HI_CHAR *gestureName = NULL;
    switch (resBuf.num) {
        case 0u:
            gestureName = "gesture fist";
           
            SAMPLE_PRT("----gesture name----:%s\n", gestureName);
            break;
        case 1u:
            gestureName = "gesture indexUp";
         
            SAMPLE_PRT("----gesture name----:%s\n", gestureName);
            break;
        case 2u:
            gestureName = "gesture OK";
           
            SAMPLE_PRT("----gesture name----:%s\n", gestureName);
            break;
        case 3u:
            gestureName = "gesture palm";
         
            SAMPLE_PRT("----gesture name----:%s\n", gestureName);
            break;
        case 4u:
            gestureName = "gesture yes";
       
            SAMPLE_PRT("----gesture name----:%s\n", gestureName);
            break;
        case 5u:
            gestureName = "gesture pinchOpen";

            SAMPLE_PRT("----gesture name----:%s\n", gestureName);
            break;
        case 6u:
            gestureName = "gesture phoneCall";
           
            SAMPLE_PRT("----gesture name----:%s\n", gestureName);
            break;
        default:
            gestureName = "gesture others";

            SAMPLE_PRT("----gesture name----:%s\n", gestureName);
            break;
    }
    SAMPLE_PRT("hand gesture success\n");
}

/*
 * 手部检测和手势分类推理
 * Hand detect and classify calculation
 */
HI_S32 Yolo2HandDetectResnetClassifyCal(uintptr_t model, VIDEO_FRAME_INFO_S *srcFrm, VIDEO_FRAME_INFO_S *dstFrm)
{
    SAMPLE_SVP_NNIE_CFG_S *self = (SAMPLE_SVP_NNIE_CFG_S*)model;//模型
    HI_S32 resLen = 0;
    int objNum;
    int ret=0;
    int num = 0;
    //传输数据至电脑端
    // printf("\n\n2\n\n");
    client_socket = socket(AF_INET, SOCK_STREAM, 0);
if (client_socket == -1) {
        std::cout << "create failed" << std::endl;
        close(client_socket);
    }
    else{
    struct sockaddr_in server_addr{};
    //接受方IP地址
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr("192.168.200.1");
    server_addr.sin_port = htons(8888);
    if (connect(client_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
        std::cout << "send image connect failed" << std::endl;
        close(client_socket);  
    }
    else{
        // std::cout<<"\nsuccess\n";
        Mat image;
    frame2Mat(srcFrm, image);
    if (image.empty()) {
        std::cout << "image transferred failed" << std::endl;
        close(client_socket);
        return 1;
    }
else{
    // 编码图像为JPEG格式的字节流
    std::vector<uchar> encoded_image;
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_params.push_back(90);
    cv::imencode(".jpg", image, encoded_image, compression_params);

   
    if (send(client_socket, encoded_image.data(), encoded_image.size(), 0) == -1) {
        std::cout << "Send failed" << std::endl;
        close(client_socket);   
    }
        else{
            // std::cout << "image result sended OK" << std::endl;
            //关闭套接字
            close(client_socket);
        }
}
    }
    }
   

    
    // VIDEO_FRAME_INFO_S *calFrm;
    //     ret = MppFrmResize(srcFrm, calFrm, 640, 480); // 640: FRM_WIDTH, 480: FRM_HEIGHT
    
    ret = FrmToOrigImg((VIDEO_FRAME_INFO_S*)srcFrm, &img);
    SAMPLE_CHECK_EXPR_RET(ret != HI_SUCCESS, ret, "face detect for YUV Frm to Img FAIL, ret=%#x\n", ret);

    objNum = HandDetectCal(&img, objs); // Send IMG to the detection net for reasoning
    for (int i = 0; i < objNum; i++) {
        cnnBoxs[i] = objs[i].box;
        RectBox *box = &objs[i].box;
        RectBoxTran(box, HAND_FRM_WIDTH, HAND_FRM_HEIGHT,
            dstFrm->stVFrame.u32Width, dstFrm->stVFrame.u32Height);
        // SAMPLE_PRT("yolo2_out: {%d, %d, %d, %d}\n", box->xmin, box->ymin, box->xmax, box->ymax);
        boxs[i] = *box;
    }
    biggestBoxIndex = GetBiggestHandIndex(boxs, objNum);
    // SAMPLE_PRT("biggestBoxIndex:%d, objNum:%d\n", biggestBoxIndex, objNum);

    /*
     * 当检测到对象时，在DSTFRM中绘制一个矩形
     * When an object is detected, a rectangle is drawn in the DSTFRM
     */
    if (biggestBoxIndex >= 0) {
        objBoxs[0] = boxs[biggestBoxIndex];
        MppFrmDrawRects(dstFrm, objBoxs, 1, RGB888_BLACK, DRAW_RETC_THICK); // Target hand objnum is equal to 1
        // printf("draw Green line\n");
        for (int j = 0; (j < objNum) && (objNum > 1); j++) {
            if (j != biggestBoxIndex) {
                remainingBoxs[num++] = boxs[j];
                /*
                 * 其他手objnum等于objnum -1
                 * Others hand objnum is equal to objnum -1
                 */
                MppFrmDrawRects(dstFrm, remainingBoxs, objNum - 1, RGB888_RED, DRAW_RETC_THICK);
            }
        }

        /*
         * 裁剪出来的图像通过预处理送分类网进行推理
         * The cropped image is preprocessed and sent to the classification network for inference
         */
        //不需要分类
        // ret = ImgYuvCrop(&img, &imgIn, &cnnBoxs[biggestBoxIndex]);
        // SAMPLE_CHECK_EXPR_RET(ret < 0, ret, "ImgYuvCrop FAIL, ret=%#x\n", ret);

        // if ((imgIn.u32Width >= WIDTH_LIMIT) && (imgIn.u32Height >= HEIGHT_LIMIT)) {
        //     COMPRESS_MODE_E enCompressMode = srcFrm->stVFrame.enCompressMode;
        //     ret = OrigImgToFrm(&imgIn, &frmIn);
        //     frmIn.stVFrame.enCompressMode = enCompressMode;
        //     SAMPLE_PRT("crop u32Width = %d, img.u32Height = %d\n", imgIn.u32Width, imgIn.u32Height);
        //     ret = MppFrmResize(&frmIn, &frmDst, IMAGE_WIDTH, IMAGE_HEIGHT);
        //     ret = FrmToOrigImg(&frmDst, &imgDst);
        //     ret = CnnCalImg(self,  &imgDst, numInfo, sizeof(numInfo) / sizeof((numInfo)[0]), &resLen);
        //     SAMPLE_CHECK_EXPR_RET(ret < 0, ret, "CnnCalImg FAIL, ret=%#x\n", ret);
        //     HI_ASSERT(resLen <= sizeof(numInfo) / sizeof(numInfo[0]));
        //     HandDetectFlag(numInfo[0]);
        //     MppFrmDestroy(&frmDst);
        // }
        // IveImgDestroy(&imgIn);
    }

    return ret;
}
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */

