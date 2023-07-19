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

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <unistd.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/prctl.h>
#include "sample_comm_ive.h"
#include "hi_debug.h"
#include "hi_comm_ive.h"
#include<opencv2/opencv.hpp>
#include "hi_mipi_tx.h"
#include "sdk.h"
#include "sample_comm.h"
#include "ai_infer_process.h"
#include "tennis_detect.h"
#include "vgs_img.h"
#include "base_interface.h"
#include "posix_help.h"
#include "sample_media_ai.h"
#include "sample_media_opencv.h"
#include "smp_color_space_convert.h"
#include "sample_comm_ive.h"

using namespace std;
using namespace cv;
#define IMG_FULL_CHN    3 // Full channel / three channel, for YUV444, RGB888
#define IMG_HALF_CHN    2 // Half channel, for YUV420/422
#define THREE_TIMES     3
#define TWO_TIMES       2

/*
 * 常用数值单位
 * Commonly used numerical units
 */
#define HI_KB               1024
#define HI_MB               (1024 * 1024)
#define HI_MS_OF_SEC        1000 // 1s in milliseconds
#define HI_NS_OF_MS         1000000 // Nanoseconds in 1ms
#define HI_BYTE_BITS        8 // Number of bits in 1 byte
#define HI_OVEN_BASE        2 // Even base
#define HI_INT8_BITS        8 // 8-bit integer number of bits
#define HI_INT16_BITS       16 // 16-bit integer number of bits
#define HI_INT32_BITS       32 // 32-bit integer number of bits
#define HI_INT64_BITS       64 // The number of bits of a 64-bit integer
#define HI_PER_BASE         100

/*
 * 调试log等级
 * Debug log level
 */
#define HI_DLEV_NONE        0 // disable
#define HI_DLEV_ERR         1 // error
#define HI_DLEV_WARN        2 // warning
#define HI_DLEV_INFO        3 // informational
#define HI_DLEV_DBG         4 // debug normal
#define HI_DLEV_VERB        5 // debug vervose
#define HI_DLEV_BUTT        6

#define LOGI(format, ...) LOG_ONLY(HI_DLEV_INFO, format, ##__VA_ARGS__)

/*
 * 打印log文件格式
 * Log with file/name
 */
#define LOG_ONLY(lev, format, ...) do { \
    if (g_hiDbgLev >= (lev)) { \
        printf(format, ##__VA_ARGS__); \
    } \
}   while (0)

/*
 * 矩形坐标结构体定义
 * Rectangular coordinate structure definition
 */

/*
 * 对齐类型
 * Alignment type
 */


typedef struct HiSampleIveColorSpaceConvInfo {
    IVE_SRC_IMAGE_S stSrc;
    FILE* pFpSrc;
    FILE* pFpDst;
} SampleIveColorSpaceConvInfo;

static SampleIveColorSpaceConvInfo g_stColorSpaceInfo;

/*
 * 调试等级
 * Debug level
 */
int g_hiDbgLev = HI_DLEV_INFO;
int IveImgCreate(IVE_IMAGE_S* img,
    IVE_IMAGE_TYPE_E enType, uint32_t width, uint32_t height);



int FrmToU8c1Img(const VIDEO_FRAME_INFO_S* frm, IVE_IMAGE_S *img);
int OrigImgToFrm(const IVE_IMAGE_S *img, VIDEO_FRAME_INFO_S* frm);
int IveImgCreate(IVE_IMAGE_S* img,
    IVE_IMAGE_TYPE_E enType, uint32_t width, uint32_t height);
void IveImgDestroy(IVE_IMAGE_S* img);
int HiAlign16(int num)
{
    return (((num) + 16 - 1) / 16 * 16); // 16: Align16
}
int ImgRgbToYuv(IVE_IMAGE_S *src, IVE_IMAGE_S *dst, IVE_IMAGE_TYPE_E dstType)
{
    IVE_HANDLE iveHnd;
    HI_BOOL finish;
    HI_S32 ret;

    if (memset_s(dst, sizeof(*dst), 0, sizeof(*dst)) != EOK) {
        HI_ASSERT(0);
    }

    ret = IveImgCreate(dst, dstType, src->u32Width, src->u32Height);
    SAMPLE_CHECK_EXPR_RET(HI_SUCCESS != ret, ret, "Error(%#x), IveImgCreate failed!\n", ret);

    IVE_CSC_CTRL_S stCSCCtrl = { IVE_CSC_MODE_VIDEO_BT601_RGB2YUV};
    ret = HI_MPI_IVE_CSC(&iveHnd, src, dst, &stCSCCtrl, HI_TRUE);
    SAMPLE_CHECK_EXPR_RET(HI_SUCCESS != ret, ret, "Error(%#x), HI_MPI_IVE_CSC failed!\n", ret);

    ret = HI_MPI_IVE_Query(iveHnd, &finish, HI_TRUE);
    SAMPLE_CHECK_EXPR_RET(HI_SUCCESS != ret, ret, "Error(%#x), HI_MPI_IVE_Query failed!\n", ret);
    return ret;
}

int HiAlign32(int num)
{
    return (((num) + 32 - 1) / 32 * 32); // 32: Align32
}

/*
 * 取路径的文件名部分
 * Take the file name part of the path
 */
const char* HiPathName(const char* path)
{
    HI_ASSERT(path);

    const char *p = strrchr(path, '/');
    if (p) {
        return p + 1;
    }
    return path;
}

/*
 * 计算通道的步幅
 * Calculate the stride of a channel
 */
static uint32_t IveCalStride(IVE_IMAGE_TYPE_E enType, uint32_t width, AlignType align)
{
    uint32_t size = 1;

    switch (enType) {
        case IVE_IMAGE_TYPE_U8C1:
        case IVE_IMAGE_TYPE_S8C1:
        case IVE_IMAGE_TYPE_S8C2_PACKAGE:
        case IVE_IMAGE_TYPE_S8C2_PLANAR:
        case IVE_IMAGE_TYPE_U8C3_PACKAGE:
        case IVE_IMAGE_TYPE_U8C3_PLANAR:
            size = sizeof(HI_U8);
            break;
        case IVE_IMAGE_TYPE_S16C1:
        case IVE_IMAGE_TYPE_U16C1:
            size = sizeof(HI_U16);
            break;
        case IVE_IMAGE_TYPE_S32C1:
        case IVE_IMAGE_TYPE_U32C1:
            size = sizeof(uint32_t);
            break;
        case IVE_IMAGE_TYPE_S64C1:
        case IVE_IMAGE_TYPE_U64C1:
            size = sizeof(uint64_t);
            break;
        default:
            break;
    }

    if (align == ALIGN_TYPE_16) {
        return HiAlign16(width * size);
    } else if (align == ALIGN_TYPE_32) {
        return HiAlign32(width * size);
    } else {
        HI_ASSERT(0);
        return 0;
    }
}

/*
 * 根据类型和大小创建缓存
 * Create IVE image buffer based on type and size
 */


/*
 * 销毁IVE image
 * Destory IVE image
 */


/*
 * 函数：色彩转换去初始化
 * function : color convert uninit
 */
static HI_VOID SampleIveColorConvertUninit(SampleIveColorSpaceConvInfo* pstColorConvertInfo)
{
    IveImgDestroy(&pstColorConvertInfo->stSrc);

    IVE_CLOSE_FILE(pstColorConvertInfo->pFpSrc);
    IVE_CLOSE_FILE(pstColorConvertInfo->pFpDst);
}

/*
 * 函数：色彩转换初始化
 * function : color convert init
 */
static HI_S32 SampleIveColorConvertInit(SampleIveColorSpaceConvInfo* g_stColorSpaceInfo,
    HI_CHAR* pchSrcFileName, HI_CHAR* pchDstFileName, HI_U32 u32Width, HI_U32 u32Height)
{
    HI_S32 s32Ret;

    memset_s(g_stColorSpaceInfo, sizeof(SampleIveColorSpaceConvInfo), 0, sizeof(SampleIveColorSpaceConvInfo));

    s32Ret = SAMPLE_COMM_IVE_CreateImage(&g_stColorSpaceInfo->stSrc, IVE_IMAGE_TYPE_YUV420SP, u32Width, u32Height);
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, COLOR_CONVERT_INIT_FAIL,
        "Error(%#x), Create Src Image failed!\n", s32Ret);

    s32Ret = HI_FAILURE;
    g_stColorSpaceInfo->pFpSrc = fopen(pchSrcFileName, "rb");
    SAMPLE_CHECK_EXPR_GOTO(HI_NULL == g_stColorSpaceInfo->pFpSrc, COLOR_CONVERT_INIT_FAIL,
        "Error, Open file %s failed!\n", pchSrcFileName);

    g_stColorSpaceInfo->pFpDst = fopen(pchDstFileName, "wb");
    SAMPLE_CHECK_EXPR_GOTO(HI_NULL == g_stColorSpaceInfo->pFpDst, COLOR_CONVERT_INIT_FAIL,
        "Error, Open file %s failed!\n", pchDstFileName);

    s32Ret = HI_SUCCESS;

COLOR_CONVERT_INIT_FAIL:

    if (HI_SUCCESS != s32Ret) {
        SampleIveColorConvertUninit(g_stColorSpaceInfo);
    }
    return s32Ret;
}

static HI_S32 SampleIveReadFile(SampleIveColorSpaceConvInfo* g_stColorSpaceInfo)
{
    HI_S32 s32Ret = SAMPLE_COMM_IVE_ReadFile(&(g_stColorSpaceInfo->stSrc), g_stColorSpaceInfo->pFpSrc);
    SAMPLE_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, "Error(%#x), Read src file failed!\n", s32Ret);
    return s32Ret;
}

static HI_BOOL s_bOpenCVProcessStopSignal = HI_FALSE;
static pthread_t g_openCVProcessThread = 0;
static int g_opencv = 0;
static AicMediaInfo g_aicTennisMediaInfo = { 0 };
static AiPlugLib g_tennisWorkPlug = {0};
static HI_CHAR tennisDetectThreadName[16] = {0};
int ImgRgbToBgr(IVE_IMAGE_S *img)
{
    uint8_t *rp = NULL;
    uint8_t *bp = NULL;
    uint8_t c;
    int i, j;

    HI_ASSERT(img->enType == IVE_IMAGE_TYPE_U8C3_PLANAR);
    HI_ASSERT(img->au32Stride[0] >= img->u32Width);
    HI_ASSERT(img->au32Stride[1] >= img->u32Width);
    HI_ASSERT(img->au32Stride[2] >= img->u32Width); // 2: au32Stride array subscript, not out of bounds

    rp = (uint8_t*)(uintptr_t)img->au64VirAddr[0];
    bp = (uint8_t*)(uintptr_t)img->au64VirAddr[2]; // 2: VirAddr array subscript, not out of bounds
    HI_ASSERT(rp && bp);
    for (i = 0; i < img->u32Height; i++) {
        for (j = 0; j < img->u32Width; j++) {
            c = rp[j];
            rp[j] = bp[j];
            bp[j] = c;
        }
        rp += img->au32Stride[0];
        bp += img->au32Stride[2]; // 2: au32Stride array subscript, not out of bounds
    }
    return 0;
}

/*
 * VIDEO_FRAME_INFO_S格式转换成IVE_IMAGE_S格式
 * 复制数据指针，不复制数据
 *
 * Video frame to IVE image.
 * Copy the data pointer, do not copy the data.
 */
int FrmToU8c1Img(const VIDEO_FRAME_INFO_S* frm, IVE_IMAGE_S *img)
{
    PIXEL_FORMAT_E pixelFormat = frm->stVFrame.enPixelFormat;

    if (memset_s(img, sizeof(*img), 0, sizeof(*img)) != EOK) {
        HI_ASSERT(0);
    }
    if (pixelFormat != PIXEL_FORMAT_YVU_SEMIPLANAR_420 &&
        pixelFormat == PIXEL_FORMAT_YVU_SEMIPLANAR_422) {
        LOGI("FrmToU8c1Img() only supp yuv420sp/yuv422sp\n");
        HI_ASSERT(0);
        return -1;
    }

    img->enType = IVE_IMAGE_TYPE_U8C1;
    img->u32Width = frm->stVFrame.u32Width;
    img->u32Height = frm->stVFrame.u32Height;

    img->au64PhyAddr[0] = frm->stVFrame.u64PhyAddr[0];
    img->au64VirAddr[0] = frm->stVFrame.u64VirAddr[0];
    img->au32Stride[0] = frm->stVFrame.u32Stride[0];
    return 0;
}

/*
 * YUV VIDEO_FRAME_INFO_S格式转成RGB IVE_DST_IMAGE_S格式
 * YUV video frame to RGB IVE image
 */
int FrmToRgbImg(VIDEO_FRAME_INFO_S* srcFrm, IVE_DST_IMAGE_S *dstImg)
{
    HI_ASSERT(srcFrm && dstImg);
    const static int chnNum = 3;
    IVE_HANDLE iveHnd;
    IVE_SRC_IMAGE_S srcImg;
    HI_BOOL finish;
    int ret;

    if (memset_s(&srcImg, sizeof(srcImg), 0, sizeof(srcImg)) != EOK) {
        HI_ASSERT(0);
    }
    srcImg.u32Width = srcFrm->stVFrame.u32Width;
    srcImg.u32Height = srcFrm->stVFrame.u32Height;

    PIXEL_FORMAT_E enPixelFormat = srcFrm->stVFrame.enPixelFormat;
    if (enPixelFormat == PIXEL_FORMAT_YVU_SEMIPLANAR_420) {
        srcImg.enType = IVE_IMAGE_TYPE_YUV420SP;
    } else if (enPixelFormat == PIXEL_FORMAT_YVU_SEMIPLANAR_422) {
        srcImg.enType = IVE_IMAGE_TYPE_YUV422SP;
    } else {
        HI_ASSERT(0);
        return -1;
    }

    /*
     * 分别复制3个通道的地址
     * Copy the addresses of the 3 channels separately
     */
    for (int i = 0; i < chnNum; i++) {
        srcImg.au64PhyAddr[i] = srcFrm->stVFrame.u64PhyAddr[i];
        srcImg.au64VirAddr[i] = srcFrm->stVFrame.u64VirAddr[i];
        srcImg.au32Stride[i] = srcFrm->stVFrame.u32Stride[i];
    }

    ret = IveImgCreate(dstImg, IVE_IMAGE_TYPE_U8C3_PLANAR, srcImg.u32Width, srcImg.u32Height);
    SAMPLE_CHECK_EXPR_RET(HI_SUCCESS != ret, ret, "Error(%#x), IveImgCreate failed!\n", ret);

    IVE_CSC_CTRL_S stCSCCtrl = { IVE_CSC_MODE_PIC_BT601_YUV2RGB};
    ret = HI_MPI_IVE_CSC(&iveHnd, &srcImg, dstImg, &stCSCCtrl, HI_TRUE);
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != ret, FAIL, "Error(%#x), HI_MPI_IVE_CSC failed!\n", ret);

    ret = HI_MPI_IVE_Query(iveHnd, &finish, HI_TRUE);
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != ret, FAIL, "Error(%#x), HI_MPI_IVE_Query failed!\n", ret);
    return ret;

    FAIL:
        IveImgDestroy(dstImg);
        return ret;
}

/*
 * 通过IVE将RGB格式转成YUV格式
 * IVE image RGB to YUV
 */

void SampleIveOrigImgToFrm(void)
{
    HI_U16 u32Width = 1920;
    HI_U16 u32Height = 1080;

    HI_CHAR* pchSrcFileName = "./userdata/data/input/color_convert_img/UsePic_1920_1080_420.yuv";
    HI_CHAR achDstFileName[IVE_FILE_NAME_LEN];
    VIDEO_FRAME_INFO_S frm;
    HI_S32 s32Ret;
    /*
     * 初始化g_stColorSpaceInfo结构体
     * Initialize the g_stColorSpaceInfo structure
     */
    memset_s(&g_stColorSpaceInfo, sizeof(g_stColorSpaceInfo), 0, sizeof(g_stColorSpaceInfo));
    SAMPLE_COMM_IVE_CheckIveMpiInit();

    if (snprintf_s(achDstFileName, sizeof(achDstFileName), sizeof(achDstFileName) - 1,
        "./userdata/data/output/color_convert_res/complete_%s.yuv", "ive_to_video") < 0) {
        HI_ASSERT(0);
    }

    s32Ret = SampleIveColorConvertInit(&g_stColorSpaceInfo, pchSrcFileName, achDstFileName, u32Width, u32Height);
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CONVERT_FAIL,
        "Error(%#x), SampleIveColorConvertInit failed!\n", s32Ret);

    s32Ret = SampleIveReadFile(&g_stColorSpaceInfo);
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CONVERT_FAIL, "Error(%#x), SampleIveReadFile failed!\n", s32Ret);

    s32Ret = OrigImgToFrm(&g_stColorSpaceInfo.stSrc, &frm);
    LOGI("IVE_IMAGE_S====width:%d, height:%d, entype:%d\n",
        g_stColorSpaceInfo.stSrc.u32Width, g_stColorSpaceInfo.stSrc.u32Height, g_stColorSpaceInfo.stSrc.enType);
    LOGI("VIDEO_FRAME_INFO_S====width:%d, height:%d, entype:%d\n",
        frm.stVFrame.u32Width, frm.stVFrame.u32Height, frm.stVFrame.enPixelFormat);
    LOGI("congratulate origImgToFrm success\n");
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CONVERT_FAIL, "Error(%#x), OrigImgToFrm failed!\n", s32Ret);

    memset_s(&frm, sizeof(VIDEO_FRAME_INFO_S), 0, sizeof(VIDEO_FRAME_INFO_S));
    SampleIveColorConvertUninit(&g_stColorSpaceInfo);
    memset_s(&g_stColorSpaceInfo, sizeof(g_stColorSpaceInfo), 0, sizeof(g_stColorSpaceInfo));

CONVERT_FAIL:
    SAMPLE_COMM_IVE_IveMpiExit();
}

/*
 * Sample实现将video frame格式转成IVE image格式
 * Sample implements converting video frame format into IVE image format
 */
void SampleIveFrmToOrigImg(void)
{
    HI_U16 u32Width = 1920;
    HI_U16 u32Height = 1080;

    HI_CHAR* pchSrcFileName = "./userdata/data/input/color_convert_img/UsePic_1920_1080_420.yuv";
    HI_CHAR achDstFileName[IVE_FILE_NAME_LEN];
    VIDEO_FRAME_INFO_S frm;
    IVE_IMAGE_S img;
    HI_S32 s32Ret;
    /*
     * 初始化g_stColorSpaceInfo结构体
     * Initialize the g_stColorSpaceInfo structure
     */
    memset_s(&g_stColorSpaceInfo, sizeof(g_stColorSpaceInfo), 0, sizeof(g_stColorSpaceInfo));
    SAMPLE_COMM_IVE_CheckIveMpiInit();

    if (snprintf_s(achDstFileName, sizeof(achDstFileName), sizeof(achDstFileName) - 1,
        "./userdata/data/output/color_convert_res/complete_%s.yuv", "frm_orig_img") < 0) {
        HI_ASSERT(0);
    }

    s32Ret = SampleIveColorConvertInit(&g_stColorSpaceInfo, pchSrcFileName, achDstFileName, u32Width, u32Height);
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CONVERT_FAIL,
        "Error(%#x), SampleIveColorConvertInit failed!\n", s32Ret);

    s32Ret = SampleIveReadFile(&g_stColorSpaceInfo);
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CONVERT_FAIL, "Error(%#x), SampleIveReadFile failed!\n", s32Ret);

    s32Ret = OrigImgToFrm(&g_stColorSpaceInfo.stSrc, &frm);
    LOGI("VIDEO_FRAME_INFO_S====width:%d, height:%d, entype:%d\n",
        frm.stVFrame.u32Width, frm.stVFrame.u32Height, frm.stVFrame.enPixelFormat);
    LOGI("IVE_IMAGE_S====width:%d, height:%d, entype:%d\n",
        g_stColorSpaceInfo.stSrc.u32Width, g_stColorSpaceInfo.stSrc.u32Height, g_stColorSpaceInfo.stSrc.enType);
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CONVERT_FAIL, "Error(%#x), OrigImgToFrm failed!\n", s32Ret);

    s32Ret = FrmToOrigImg(&frm, &img);
    LOGI("IVE_IMAGE_S====width:%d, height:%d, entype:%d\n", img.u32Width, img.u32Height, img.enType);
    LOGI("VIDEO_FRAME_INFO_S====width:%d, height:%d, entype:%d\n",
        frm.stVFrame.u32Width, frm.stVFrame.u32Height, frm.stVFrame.enPixelFormat);
    LOGI("congratulate FrmToOrigImg success\n");
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CONVERT_FAIL, "Error(%#x), FrmToOrigImg failed!\n", s32Ret);

    s32Ret = SAMPLE_COMM_IVE_WriteFile(&img, g_stColorSpaceInfo.pFpDst);
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CONVERT_FAIL,
        "Error(%#x), SAMPLE_COMM_IVE_WriteFile failed!\n", s32Ret);

    memset_s(&frm, sizeof(VIDEO_FRAME_INFO_S), 0, sizeof(VIDEO_FRAME_INFO_S));
    memset_s(&img, sizeof(IVE_IMAGE_S), 0, sizeof(IVE_IMAGE_S));
    SampleIveColorConvertUninit(&g_stColorSpaceInfo);
    memset_s(&g_stColorSpaceInfo, sizeof(g_stColorSpaceInfo), 0, sizeof(g_stColorSpaceInfo));

CONVERT_FAIL:
    SAMPLE_COMM_IVE_IveMpiExit();
}

/*
 * Sample实现将video frame格式转成U8C1格式
 * Sample implements converting video frame format into U8C1 format
 */
void SampleIveFrmToU8c1Img(void)
{
    HI_U16 u32Width = 1920;
    HI_U16 u32Height = 1080;

    HI_CHAR* pchSrcFileName = "./userdata/data/input/color_convert_img/UsePic_1920_1080_420.yuv";
    HI_CHAR achDstFileName[IVE_FILE_NAME_LEN];
    VIDEO_FRAME_INFO_S frm;
    IVE_IMAGE_S img;
    HI_S32 s32Ret;
    /*
     * 初始化g_stColorSpaceInfo结构体
     * Initialize the g_stColorSpaceInfo structure
     */
    memset_s(&g_stColorSpaceInfo, sizeof(g_stColorSpaceInfo), 0, sizeof(g_stColorSpaceInfo));
    SAMPLE_COMM_IVE_CheckIveMpiInit();

    if (snprintf_s(achDstFileName, sizeof(achDstFileName), sizeof(achDstFileName) - 1,
        "./userdata/data/output/color_convert_res/complete_%s.yuv", "u8c1") < 0) {
        HI_ASSERT(0);
    };

    s32Ret = SampleIveColorConvertInit(&g_stColorSpaceInfo, pchSrcFileName, achDstFileName, u32Width, u32Height);
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CONVERT_FAIL,
        "Error(%#x), SampleIveColorConvertInit failed!\n", s32Ret);

    s32Ret = SampleIveReadFile(&g_stColorSpaceInfo);
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CONVERT_FAIL, "Error(%#x), SampleIveReadFile failed!\n", s32Ret);

    s32Ret = OrigImgToFrm(&g_stColorSpaceInfo.stSrc, &frm);
    LOGI("VIDEO_FRAME_INFO_S====width:%d, height:%d, entype:%d\n",
        frm.stVFrame.u32Width, frm.stVFrame.u32Height, frm.stVFrame.enPixelFormat);
    LOGI("IVE_IMAGE_S====width:%d, height:%d, entype:%d\n",
        g_stColorSpaceInfo.stSrc.u32Width, g_stColorSpaceInfo.stSrc.u32Height, g_stColorSpaceInfo.stSrc.enType);
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CONVERT_FAIL, "Error(%#x), OrigImgToFrm failed!\n", s32Ret);

    s32Ret = FrmToU8c1Img(&frm, &img);
    LOGI("VIDEO_FRAME_INFO_S====width:%d, height:%d, entype:%d\n",
        frm.stVFrame.u32Width, frm.stVFrame.u32Height, frm.stVFrame.enPixelFormat);
    LOGI("IVE_IMAGE_S====width:%d, height:%d, entype:%d\n", img.u32Width, img.u32Height, img.enType);
    LOGI("congratulate FrmToU8c1Img success\n");
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CONVERT_FAIL, "Error(%#x), FrmToU8c1Img failed!\n", s32Ret);

    s32Ret = SAMPLE_COMM_IVE_WriteFile(&img, g_stColorSpaceInfo.pFpDst);
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CONVERT_FAIL,
        "Error(%#x), SAMPLE_COMM_IVE_WriteFile failed!\n", s32Ret);

    memset_s(&frm, sizeof(VIDEO_FRAME_INFO_S), 0, sizeof(VIDEO_FRAME_INFO_S));
    memset_s(&img, sizeof(IVE_IMAGE_S), 0, sizeof(IVE_IMAGE_S));
    SampleIveColorConvertUninit(&g_stColorSpaceInfo);
    memset_s(&g_stColorSpaceInfo, sizeof(g_stColorSpaceInfo), 0, sizeof(g_stColorSpaceInfo));

CONVERT_FAIL:
    SAMPLE_COMM_IVE_IveMpiExit();
}

/*
 * Sample实现将video frame格式转成RGB格式，最后转成YUV格式进行存储
 * Sample realizes converting the video frame format into RGB format, and finally into YUV format for storage
 */
void SampleIveFrmToRgbImgToYuv(void)
{
    HI_U16 u32Width = 1920;
    HI_U16 u32Height = 1080;

    HI_CHAR* pchSrcFileName = "./userdata/data/input/color_convert_img/UsePic_1920_1080_420.yuv";
    HI_CHAR achDstFileName[IVE_FILE_NAME_LEN];
    VIDEO_FRAME_INFO_S frm;
    IVE_IMAGE_S img;
    IVE_IMAGE_S dst;
    HI_S32 s32Ret;
    /*
     * 初始化g_stColorSpaceInfo结构体
     * Initialize the g_stColorSpaceInfo structure
     */
    memset_s(&g_stColorSpaceInfo, sizeof(g_stColorSpaceInfo), 0, sizeof(g_stColorSpaceInfo));
    SAMPLE_COMM_IVE_CheckIveMpiInit();

    if (snprintf_s(achDstFileName, sizeof(achDstFileName), sizeof(achDstFileName) - 1,
        "./userdata/data/output/color_convert_res/complete_%s.yuv", "frm_rgb_yuv") < 0) {
        HI_ASSERT(0);
    }

    s32Ret = SampleIveColorConvertInit(&g_stColorSpaceInfo, pchSrcFileName, achDstFileName, u32Width, u32Height);
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CONVERT_FAIL,
        "Error(%#x), SampleIveColorConvertInit failed!\n", s32Ret);

    s32Ret = SampleIveReadFile(&g_stColorSpaceInfo);
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CONVERT_FAIL, "Error(%#x), SampleIveReadFile failed!\n", s32Ret);

    s32Ret = OrigImgToFrm(&g_stColorSpaceInfo.stSrc, &frm);
    LOGI("VIDEO_FRAME_INFO_S====width:%d, height:%d, entype:%d\n",
        frm.stVFrame.u32Width, frm.stVFrame.u32Height, frm.stVFrame.enPixelFormat);
    LOGI("IVE_IMAGE_S====width:%d, height:%d, entype:%d\n",
        g_stColorSpaceInfo.stSrc.u32Width, g_stColorSpaceInfo.stSrc.u32Height, g_stColorSpaceInfo.stSrc.enType);
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CONVERT_FAIL, "Error(%#x), OrigImgToFrm failed!\n", s32Ret);

    s32Ret = FrmToRgbImg(&frm, &img);
    LOGI("VIDEO_FRAME_INFO_S====width:%d, height:%d, entype:%d\n",
        frm.stVFrame.u32Width, frm.stVFrame.u32Height, frm.stVFrame.enPixelFormat);
    LOGI("IVE_IMAGE_S====width:%d, height:%d, entype:%d\n", img.u32Width, img.u32Height, img.enType);
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CONVERT_FAIL, "Error(%#x), FrmToRgbImg failed!\n", s32Ret);

    s32Ret = ImgRgbToYuv(&img, &dst, IVE_IMAGE_TYPE_YUV420SP);
    LOGI("congratulate ImgRgbToYuv success\n");
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CONVERT_FAIL, "Error(%#x), ImgRgbToYuv failed!\n", s32Ret);

    s32Ret = SAMPLE_COMM_IVE_WriteFile(&dst, g_stColorSpaceInfo.pFpDst);
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CONVERT_FAIL,
        "Error(%#x), SAMPLE_COMM_IVE_WriteFile failed!\n", s32Ret);

    IveImgDestroy(&img);
    IveImgDestroy(&dst);
    memset_s(&frm, sizeof(VIDEO_FRAME_INFO_S), 0, sizeof(VIDEO_FRAME_INFO_S));
    SampleIveColorConvertUninit(&g_stColorSpaceInfo);
    memset_s(&g_stColorSpaceInfo, sizeof(g_stColorSpaceInfo), 0, sizeof(g_stColorSpaceInfo));

CONVERT_FAIL:
    SAMPLE_COMM_IVE_IveMpiExit();
}

/*
 * Sample实现将video frame格式转成RGB格式，再转成BGR格式
 * Sample implements converting the video frame format into RGB format and then into BGR format
 */
void SampleIveFrmToRgbImgToBgr(void)
{
    HI_U16 u32Width = 1920;
    HI_U16 u32Height = 1080;

    HI_CHAR* pchSrcFileName = "./userdata/data/input/color_convert_img/UsePic_1920_1080_420.yuv";
    HI_CHAR achDstFileName[IVE_FILE_NAME_LEN];
    VIDEO_FRAME_INFO_S frm;
    IVE_IMAGE_S img;
    HI_S32 s32Ret;
    /*
     * 初始化g_stColorSpaceInfo结构体
     * Initialize the g_stColorSpaceInfo structure
     */
    memset_s(&g_stColorSpaceInfo,sizeof(g_stColorSpaceInfo),0,sizeof(g_stColorSpaceInfo));
    SAMPLE_COMM_IVE_CheckIveMpiInit();

    if (snprintf_s(achDstFileName, sizeof(achDstFileName), sizeof(achDstFileName) - 1,
        "./userdata/data/output/color_convert_res/complete_%s.bgr", "rgb2bgr") < 0) {
        HI_ASSERT(0);
    };

    s32Ret = SampleIveColorConvertInit(&g_stColorSpaceInfo, pchSrcFileName, achDstFileName, u32Width, u32Height);
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CONVERT_FAIL,
        "Error(%#x), SampleIveColorConvertInit failed!\n", s32Ret);

    s32Ret = SampleIveReadFile(&g_stColorSpaceInfo);
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CONVERT_FAIL, "Error(%#x), SampleIveReadFile failed!\n", s32Ret);

    s32Ret = OrigImgToFrm(&g_stColorSpaceInfo.stSrc, &frm);
    LOGI("VIDEO_FRAME_INFO_S====width:%d, height:%d, entype:%d\n",
        frm.stVFrame.u32Width, frm.stVFrame.u32Height, frm.stVFrame.enPixelFormat);
    LOGI("IVE_IMAGE_S====width:%d, height:%d, entype:%d\n",
        g_stColorSpaceInfo.stSrc.u32Width, g_stColorSpaceInfo.stSrc.u32Height, g_stColorSpaceInfo.stSrc.enType);
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CONVERT_FAIL, "Error(%#x), OrigImgToFrm failed!\n", s32Ret);

    s32Ret = FrmToRgbImg(&frm, &img);
    LOGI("VIDEO_FRAME_INFO_S====width:%d, height:%d, entype:%d\n",
        frm.stVFrame.u32Width, frm.stVFrame.u32Height, frm.stVFrame.enPixelFormat);
    LOGI("IVE_IMAGE_S====width:%d, height:%d, entype:%d\n", img.u32Width, img.u32Height, img.enType);
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CONVERT_FAIL, "Error(%#x), FrmToRgbImg failed!\n", s32Ret);

    s32Ret = ImgRgbToBgr(&img);
    LOGI("congratulate ImgRgbToBgr success\n");
    SAMPLE_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CONVERT_FAIL, "Error(%#x), ImgRgbToBgr failed!\n", s32Ret);

    IveImgDestroy(&img);
    memset_s(&frm, sizeof(VIDEO_FRAME_INFO_S), 0, sizeof(VIDEO_FRAME_INFO_S));
    SampleIveColorConvertUninit(&g_stColorSpaceInfo);
    memset_s(&g_stColorSpaceInfo, sizeof(g_stColorSpaceInfo), 0, sizeof(g_stColorSpaceInfo));

CONVERT_FAIL:
    SAMPLE_COMM_IVE_IveMpiExit();
}


/*
 * IVE_IMAGE_S格式转成VIDEO_FRAME_INFO_S格式
 * 复制数据指针，不复制数据
 *
 * IVE image to video frame.
 * Copy the data pointer, do not copy the data.
 */

/*
 * ����VI�豸��Ϣ
 * Set VI device information
 */
static void TennisViCfgSetDev(ViCfg* self, int devId, WDR_MODE_E wdrMode)
{
    HI_ASSERT(self);
    HI_ASSERT((int)wdrMode < WDR_MODE_BUTT);

    self->astViInfo[0].stDevInfo.ViDev = devId;
    self->astViInfo[0].stDevInfo.enWDRMode = wdrMode;
}

/*
 * ����VIͨ��
 * Set up the VI channel
 */
static void TennisViCfgSetChn(ViCfg* self, int chnId, PIXEL_FORMAT_E pixFormat,
    VIDEO_FORMAT_E videoFormat, DYNAMIC_RANGE_E dynamicRange)
{
    HI_ASSERT(self);
    HI_ASSERT((int)pixFormat < PIXEL_FORMAT_BUTT);
    HI_ASSERT((int)videoFormat < VIDEO_FORMAT_BUTT);
    HI_ASSERT((int)dynamicRange < DYNAMIC_RANGE_BUTT);
    self->astViInfo[0].stChnInfo.ViChn = chnId;
    self->astViInfo[0].stChnInfo.enPixFormat = pixFormat;
    self->astViInfo[0].stChnInfo.enVideoFormat = videoFormat;
    self->astViInfo[0].stChnInfo.enDynamicRange = dynamicRange;
}

static HI_VOID TennisViPramCfg(HI_VOID)
{
    ViCfgInit(&g_aicTennisMediaInfo.viCfg);
    TennisViCfgSetDev(&g_aicTennisMediaInfo.viCfg, 0, WDR_MODE_NONE);
    ViCfgSetPipe(&g_aicTennisMediaInfo.viCfg, 0, -1, -1, -1);
    g_aicTennisMediaInfo.viCfg.astViInfo[0].stPipeInfo.enMastPipeMode = VI_OFFLINE_VPSS_OFFLINE;
    TennisViCfgSetChn(&g_aicTennisMediaInfo.viCfg, 0, PIXEL_FORMAT_YVU_SEMIPLANAR_420,
        VIDEO_FORMAT_LINEAR, DYNAMIC_RANGE_SDR8);
    g_aicTennisMediaInfo.viCfg.astViInfo[0].stChnInfo.enCompressMode = COMPRESS_MODE_SEG;
}


static HI_VOID TennisStVbParamCfg(VbCfg *self)
{
    memset_s(&g_aicTennisMediaInfo.vbCfg, sizeof(VB_CONFIG_S), 0, sizeof(VB_CONFIG_S));
    // 2: The number of buffer pools that can be accommodated in the entire system
    self->u32MaxPoolCnt              = 2;

    /*
     * ��ȡһ֡ͼƬ��buffer��С
     * Get picture buffer size
     */
    g_aicTennisMediaInfo.u32BlkSize = COMMON_GetPicBufferSize(g_aicTennisMediaInfo.stSize.u32Width,
        g_aicTennisMediaInfo.stSize.u32Height, SAMPLE_PIXEL_FORMAT, DATA_BITWIDTH_8, COMPRESS_MODE_SEG, DEFAULT_ALIGN);
    self->astCommPool[0].u64BlkSize  = g_aicTennisMediaInfo.u32BlkSize;
    // 10: Number of cache blocks per cache pool. Value range: (0, 10240]
    self->astCommPool[0].u32BlkCnt   = 10;

    /*
     * ��ȡraw buffer�Ĵ�С
     * Get raw buffer size
     */
    g_aicTennisMediaInfo.u32BlkSize = VI_GetRawBufferSize(g_aicTennisMediaInfo.stSize.u32Width,
        g_aicTennisMediaInfo.stSize.u32Height, PIXEL_FORMAT_RGB_BAYER_16BPP, COMPRESS_MODE_NONE, DEFAULT_ALIGN);
    self->astCommPool[1].u64BlkSize  = g_aicTennisMediaInfo.u32BlkSize;
    // 4: Number of cache blocks per cache pool. Value range: (0, 10240]
    self->astCommPool[1].u32BlkCnt   = 4;
}

static HI_VOID TennisVpssParamCfg(HI_VOID)
{
    VpssCfgInit(&g_aicTennisMediaInfo.vpssCfg);
    VpssCfgSetGrp(&g_aicTennisMediaInfo.vpssCfg, 0, NULL,
        g_aicTennisMediaInfo.stSize.u32Width, g_aicTennisMediaInfo.stSize.u32Width);
    g_aicTennisMediaInfo.vpssCfg.grpAttr.enPixelFormat = PIXEL_FORMAT_YVU_SEMIPLANAR_420;
    // 1920:AICSTART_VI_OUTWIDTH, 1080: AICSTART_VI_OUTHEIGHT
    VpssCfgAddChn(&g_aicTennisMediaInfo.vpssCfg, 1, NULL, 1920, 1080);
    HI_ASSERT(!g_aicTennisMediaInfo.viSess);
}

static HI_VOID TennisStVoParamCfg(VoCfg *self)
{
    SAMPLE_COMM_VO_GetDefConfig(self);
    self->enDstDynamicRange = DYNAMIC_RANGE_SDR8;

    self->enVoIntfType = VO_INTF_MIPI; /* set VO int type */
    self->enIntfSync = VO_OUTPUT_USER; /* set VO output information */

    self->enPicSize = g_aicTennisMediaInfo.enPicSize;
}

static HI_VOID TennisDetectAiProcess(VIDEO_FRAME_INFO_S frm, VO_LAYER voLayer, VO_CHN voChn)
{
    int ret;
    tennis_detect opencv;
    if (GetCfgBool("tennis_detect_switch:support_tennis_detect", true)) {
        if (g_tennisWorkPlug.model == 0) {
            ret = opencv.TennisDetectLoad(&g_tennisWorkPlug.model);
            if (ret < 0) {
                g_tennisWorkPlug.model = 0;
                SAMPLE_CHECK_EXPR_GOTO(ret < 0, TENNIS_RELEASE, "TennisDetectLoad err, ret=%#x\n", ret);
            }
        }

        VIDEO_FRAME_INFO_S calFrm;
        ret = MppFrmResize(&frm, &calFrm, 640, 480); // 640: FRM_WIDTH, 480: FRM_HEIGHT
        ret = opencv.TennisDetectCal(g_tennisWorkPlug.model, &calFrm, &frm);
        SAMPLE_CHECK_EXPR_GOTO(ret < 0, TENNIS_RELEASE, "TennisDetectCal err, ret=%#x\n", ret);

        ret = HI_MPI_VO_SendFrame(voLayer, voChn, &frm, 0);
        SAMPLE_CHECK_EXPR_GOTO(ret != HI_SUCCESS, TENNIS_RELEASE,
            "HI_MPI_VO_SendFrame err, ret=%#x\n", ret);
        MppFrmDestroy(&calFrm);
    }

    TENNIS_RELEASE:
        ret = HI_MPI_VPSS_ReleaseChnFrame(g_aicTennisMediaInfo.vpssGrp, g_aicTennisMediaInfo.vpssChn0, &frm);
        if (ret != HI_SUCCESS) {
            SAMPLE_PRT("Error(%#x),HI_MPI_VPSS_ReleaseChnFrame failed,Grp(%d) chn(%d)!\n",
                ret, g_aicTennisMediaInfo.vpssGrp, g_aicTennisMediaInfo.vpssChn0);
        }
}

static HI_VOID* GetVpssChnFrameTennisDetect(HI_VOID* arg)
{
    int ret;
    //video_frame_info_s frm为图像
    VIDEO_FRAME_INFO_S frm;
    HI_S32 s32MilliSec = 2000;
    VO_LAYER voLayer = 0;
    VO_CHN voChn = 0;

    SAMPLE_PRT("vpssGrp:%d, vpssChn0:%d\n", g_aicTennisMediaInfo.vpssGrp, g_aicTennisMediaInfo.vpssChn0);

    while (HI_FALSE == s_bOpenCVProcessStopSignal) {
        ret = HI_MPI_VPSS_GetChnFrame(g_aicTennisMediaInfo.vpssGrp, g_aicTennisMediaInfo.vpssChn0, &frm, s32MilliSec);
        if (ret != 0) {
            
            SAMPLE_PRT("HI_MPI_VPSS_GetChnFrame FAIL, err=%#x, grp=%d, chn=%d\n",
                ret, g_aicTennisMediaInfo.vpssGrp, g_aicTennisMediaInfo.vpssChn0);
            ret = HI_MPI_VPSS_ReleaseChnFrame(g_aicTennisMediaInfo.vpssGrp, g_aicTennisMediaInfo.vpssChn0, &frm);
            if (ret != HI_SUCCESS) {
                SAMPLE_PRT("Error(%#x),HI_MPI_VPSS_ReleaseChnFrame failed,Grp(%d) chn(%d)!\n",
                    ret, g_aicTennisMediaInfo.vpssGrp, g_aicTennisMediaInfo.vpssChn0);
            }
            continue;
        }
        SAMPLE_PRT("get vpss frame success, weight:%d, height:%d\n", frm.stVFrame.u32Width, frm.stVFrame.u32Height);

        if (g_opencv == 0) {
            ConfBaseInit("./sample_ai.conf");
            g_opencv++;
        }
        TennisDetectAiProcess(frm, voLayer, voChn);
    }

    return HI_NULL;
}

static HI_VOID PauseDoUnloadTennisModel(HI_VOID)
{
    if (GetCfgBool("tennis_detect_switch:support_tennis_detect", true)) {
        memset_s(&g_tennisWorkPlug, sizeof(g_tennisWorkPlug), 0x00, sizeof(g_tennisWorkPlug));
        ConfBaseExt();
        SAMPLE_PRT("tennis detect exit success\n");
        g_opencv = 0;
    }
}

static HI_S32 TennisDetectAiThreadProcess(HI_VOID)
{
    HI_S32 s32Ret;
    if (snprintf_s(tennisDetectThreadName, sizeof(tennisDetectThreadName),
        sizeof(tennisDetectThreadName) - 1, "OpencvProcess") < 0) {
        HI_ASSERT(0);
    }
    prctl(PR_SET_NAME, (unsigned long)tennisDetectThreadName, 0, 0, 0);
    s32Ret = pthread_create(&g_openCVProcessThread, NULL, GetVpssChnFrameTennisDetect, NULL);

    return s32Ret;
}

/*
 * ��sensor�ɼ���������ʾ��Һ�����ϣ�ͬʱ�����߳�������������������
 * ��Ƶ����->��Ƶ������ϵͳ->��Ƶ���->��ʾ��
 *
 * Display the data collected by the sensor on the LCD screen,
 * and at the same time create a thread to run tennis detect reasoning calculations
 * VI->VPSS->VO->MIPI
 */
HI_S32 sample_media_opencv::SAMPLE_MEDIA_TENNIS_DETECT(HI_VOID)
{
    HI_S32 s32Ret;
    HI_S32 fd = 0;

    /*
     * ����VI����
     * Config VI parameter
     */
    TennisViPramCfg();

    /*
     * ͨ��Sensor�ͺŻ�ȡenPicSize
     * Obtain enPicSize through the Sensor type
     */
    s32Ret = SAMPLE_COMM_VI_GetSizeBySensor(g_aicTennisMediaInfo.viCfg.astViInfo[0].stSnsInfo.enSnsType,
        &g_aicTennisMediaInfo.enPicSize);
    SAMPLE_CHECK_EXPR_RET(s32Ret != HI_SUCCESS, s32Ret, "get pic size by sensor fail, s32Ret=%#x\n", s32Ret);

    /*
     * ����enPicSize���õ�ͼƬ�Ŀ���
     * Get picture size(w*h), according enPicSize
     */
    s32Ret = SAMPLE_COMM_SYS_GetPicSize(g_aicTennisMediaInfo.enPicSize, &g_aicTennisMediaInfo.stSize);
    SAMPLE_PRT("AIC: snsMaxSize=%ux%u\n", g_aicTennisMediaInfo.stSize.u32Width, g_aicTennisMediaInfo.stSize.u32Height);
    SAMPLE_CHECK_EXPR_RET(s32Ret != HI_SUCCESS, s32Ret, "get picture size failed, s32Ret=%#x\n", s32Ret);

    /*
     * ����VB����
     * Config VB parameter
     */
    TennisStVbParamCfg(&g_aicTennisMediaInfo.vbCfg);

    /*
     * ��Ƶ����س�ʼ���Լ�MPIϵͳ��ʼ��
     * VB init & MPI system init
     */
    s32Ret = SAMPLE_COMM_SYS_Init(&g_aicTennisMediaInfo.vbCfg);
    SAMPLE_CHECK_EXPR_RET(s32Ret != HI_SUCCESS, s32Ret, "system init failed, s32Ret=%#x\n", s32Ret);

    /*
     * ����VO��MIPIͨ·����ȡMIPI�豸
     * Set VO config to MIPI, get MIPI device
     */
    s32Ret = SAMPLE_VO_CONFIG_MIPI(&fd);
    SAMPLE_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, EXIT, "CONFIG MIPI FAIL.s32Ret:0x%x\n", s32Ret);

    /*
     * ����VPSS����
     * Config VPSS parameter
     */
    TennisVpssParamCfg();
    s32Ret = ViVpssCreate(&g_aicTennisMediaInfo.viSess, &g_aicTennisMediaInfo.viCfg, &g_aicTennisMediaInfo.vpssCfg);
    SAMPLE_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, EXIT1, "ViVpss Sess create FAIL, ret=%#x\n", s32Ret);
    g_aicTennisMediaInfo.vpssGrp = 0;
    g_aicTennisMediaInfo.vpssChn0 = 1;

    /*
     * ����VO����
     * Config VO parameter
     */
    TennisStVoParamCfg(&g_aicTennisMediaInfo.voCfg);

    /*
     * ����VO��MIPI lcdͨ·
     * Start VO to MIPI lcd
     */
    s32Ret = SampleCommVoStartMipi(&g_aicTennisMediaInfo.voCfg);
    SAMPLE_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, EXIT1, "start vo FAIL. s32Ret: 0x%x\n", s32Ret);

    /*
     * VPSS��VO
     * VPSS bind VO
     */
    s32Ret = SAMPLE_COMM_VPSS_Bind_VO(g_aicTennisMediaInfo.vpssGrp,
        g_aicTennisMediaInfo.vpssChn0, g_aicTennisMediaInfo.voCfg.VoDev, 0);
    SAMPLE_CHECK_EXPR_GOTO(s32Ret != HI_SUCCESS, EXIT2, "vo bind vpss FAIL. s32Ret: 0x%x\n", s32Ret);
    SAMPLE_PRT("vpssGrp:%d, vpssChn:%d\n", g_aicTennisMediaInfo.vpssGrp, g_aicTennisMediaInfo.vpssChn0);

    /*
     * ���������߳�����ai
     * Create work thread to run ai
     */
    s32Ret = TennisDetectAiThreadProcess();
    SAMPLE_CHECK_EXPR_RET(s32Ret != HI_SUCCESS, s32Ret, "ai proccess thread creat fail:%s\n", strerror(s32Ret));
    PAUSE();
    s_bOpenCVProcessStopSignal = HI_TRUE;
    /*
     * �ȴ�һ���߳̽������̼߳�ͬ���Ĳ���
     * Waiting for the end of a thread, the operation of synchronization between threads
     */
    pthread_join(g_openCVProcessThread, nullptr);
    g_openCVProcessThread = 0;
    PauseDoUnloadTennisModel();

    SAMPLE_COMM_VPSS_UnBind_VO(g_aicTennisMediaInfo.vpssGrp,
        g_aicTennisMediaInfo.vpssChn0, g_aicTennisMediaInfo.voCfg.VoDev, 0);
    SAMPLE_VO_DISABLE_MIPITx(fd);
    SampleCloseMipiTxFd(fd);
    system("echo 0 > /sys/class/gpio/gpio55/value");

EXIT2:
    SAMPLE_COMM_VO_StopVO(&g_aicTennisMediaInfo.voCfg);
EXIT1:
    VpssStop(&g_aicTennisMediaInfo.vpssCfg);
    SAMPLE_COMM_VI_UnBind_VPSS(g_aicTennisMediaInfo.viCfg.astViInfo[0].stPipeInfo.aPipe[0],
        g_aicTennisMediaInfo.viCfg.astViInfo[0].stChnInfo.ViChn, g_aicTennisMediaInfo.vpssGrp);
    ViStop(&g_aicTennisMediaInfo.viCfg);
    free(g_aicTennisMediaInfo.viSess);
EXIT:
    SAMPLE_COMM_SYS_Exit();
    return s32Ret;
}

