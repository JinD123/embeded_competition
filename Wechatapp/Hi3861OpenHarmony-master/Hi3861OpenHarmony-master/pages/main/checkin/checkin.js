const app = getApp()
let names = {};
let time = 0
Page({
  data: {
    courseOptions: ['语文课', '数学课', '英语课'],
    selectedCourse: '',
    students: [],
    stateReported:{// 添加 student_names 属性},
  }
},
  onLoad: function () {
  
    //this.generateStudents();
    this.startUpdateInterval();
 
  },
  generateStudents: function () {
 
    
    const students = names.map(name => {
      const signed = Math.random() >= 0.5;
      return {
        name: name,
        signed: signed,
      };
    });

    this.setData({
      
      students: students,
     
    });
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

        this.setData({
          stateReported: deviceData.payload.state.reported
        })
        console.log("result:", deviceData)
        this.getname()
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
  },
 startUpdateInterval:function() {
    setInterval(() => {
      this.update();
    }, 1000);
  },
  getname:function(){
 // 如果字符串以 "a" 开头，则跳转到指定程序
    let regexchar = /^a/;
      let isStartsWithA = regexchar.test(this.data.stateReported.data);
    if (isStartsWithA) {
       // 使用正则表达式匹配b找b的位置
       const find_b = this.data.stateReported.data.split(" ");
       console.log(find_b)
       let startIndex = -1; // 初始化 index 为 -1，表示没有找到字符串 b
for (let i = 0; i < find_b.length; i++) {
  if (find_b[i].indexOf('b') !== -1) { // 如果当前人名包含字符串 b
    startIndex = i; // 将 index 设置为当前位置 i
    break; // 找到后跳出循环
  }
}

console.log(startIndex); // 输出 b 出现的位置，如果没有找到则为 -1
      // 使用正则表达式匹配字符串中的数字
      const regex = /[\u4e00-\u9fa5\s]+/g;
      const matches = this.data.stateReported.data.match(regex);
      
      if (matches && matches.length > 0) {
        const names2 = matches.map(match => match.trim().split(' ')).flat();
        console.log(names2) // 输出提取到的人名数组
        const students = names2.map((name, index) => {
          // console.log(name);
          
          const signed = !(index >= startIndex);
          // console.log(signed);
          
          return {
            name: name,
            signed: signed,
          };
        });
        console.log(students)
        this.setData({
          
          students: students,
         
        });
  };
     
    } else {
      console.log('未匹配到人名');
    }
  },
  onCourseChange: function (event) {
    const { value } = event.detail;
    this.setData({
      selectedCourse: this.data.courseOptions[value],
    });
  },
});
