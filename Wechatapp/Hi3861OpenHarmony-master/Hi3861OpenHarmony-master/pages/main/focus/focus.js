// focus.js
let time=0;
let matches={};
const app = getApp()
Page({
  data: {
    courseOptions: ['语文课', '数学课', '英语课'],
    selectedCourse: '',
    focusData: [],
    stateReported: {

    },
    analysisText: '',
    analysisStyle: ''
  },
  onLoad: function() {
    this.startUpdateInterval();
  },
  
update() {


  if(time==0){
    wx.showLoading()
    
  }
  else if(time==1){
    wx.hideLoading()
    time=2
  }
  
  
  wx.cloud.callFunction({
    name: 'iothub-shadow-query',
    data: {
      ProductId: app.globalData.productId,
      DeviceName: app.globalData.deviceName,
      SecretId: app.globalData.secretId,
      SecretKey: app.globalData.secretKey,
    },
    success: res => {
      if(time==0){
        time=1
      wx.showToast({
        icon: 'none',
        title: 'Subscribe完成，获取云端数据成功',
        duration:2000
      })
    }
      let deviceData = JSON.parse(res.result.Data)
//注意正式跑的时候这个要展开
       this.setData({
         stateReported: deviceData.payload.state.reported
       })
      console.log("result:", deviceData)
     
      let regex = /^b/;
      let isStartsWithB = regex.test(this.data.stateReported.data);
      if (isStartsWithB) {
        // 使用正则表达式匹配字符串中的数字
   
       matches = this.data.stateReported.data.match(/[-+]?\d+(\.\d+)?/g);
        // 输出匹配结果
        console.log(matches);
    };
   
    },
    fail: err => {
      if(time==0){
        time=1
      wx.showToast({
        icon: 'none',
        title: 'Subscribe失败，获取云端数据失败',
        duration:2000
      })
      console.error('[云函数] [iotexplorer] 调用失败：', err)
    }
  
    }
  })
  this.generateFocusData()
  this.updateCurrentFocus()
},
startUpdateInterval:function() {
  setInterval(() => {
    this.update();
  }, 1000);
},
  generateFocusData: function() {

const focusData=[];
    const focusTypes = ['行为', '抬头', '表情', '综合'];
   
    const arr = Array.from(matches)
    const floatValues = arr.map(str => {
      const floatValue = parseFloat(str);
      return floatValue.toFixed(2);
    });
    for (let i = 0; i < focusTypes.length; i++) {
      const focusType = focusTypes[i];
      const focusValue =floatValues[i];
      const iconPath = this.generateIconPath(i + 1);
      focusData.push({
        type: focusType,
        value: focusValue,
        icon: iconPath
      });
    }
    this.setData({
      focusData: focusData
    });
    this.updateCurrentFocus();
  },
  generateIconPath: function(index) {
    return `../../../asset/imgs/${index}.png`;
  },
  // generateRandomFocusValue: function() {
    
  //   // const randomValue = Number((Math.random() * 5).toFixed(2));
  //   // return randomValue;
  // },
  onCourseChange: function(event) {
    const { value } = event.detail;
    this.setData({
      selectedCourse: this.data.courseOptions[value]
    }, () => {
      this.updateCurrentFocus();
    });
  },
  updateCurrentFocus: function() {

    const currentFocus = {
       behavior: matches[0],
       headUp:  matches[1],
       facial:  matches[2],
       overall: matches[3],
    };
    console.log('当前综合专注度数值3:', currentFocus.overall);
    let analysisText = '';
    let analysisStyle = '';
    if (currentFocus.overall < 3.5) {
      analysisText = '当前课堂实时专注度较差，请注意关注学生动态！';
      analysisStyle = 'bad';
    } else if (currentFocus.overall >= 3.5 && currentFocus.overall <= 4) {
      analysisText = '当前课堂实时专注度一般，请留意部分学生行为！';
      analysisStyle = 'average';
    } else {
      analysisText = '当前课堂实时专注度较好，请继续您的课堂吧！';
      analysisStyle = 'good';
    }
    this.setData({
      currentFocus: currentFocus,
      analysisText: analysisText,
      analysisStyle: analysisStyle
    });
  },
});
