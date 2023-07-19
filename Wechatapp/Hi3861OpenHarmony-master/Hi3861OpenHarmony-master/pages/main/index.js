// index.js
Page({
  // 页面数据
  data: {
    
  },

  /* 生命周期函数--监听页面加载 */
  onLoad: function () {
    
  },

  // 跳转到课表页面
  navigateToSchedule: function () {
    wx.navigateTo({
      url: './schedule/schedule',
    })
  },

  // 跳转到专注页面
  navigateToFocus: function () {
    wx.navigateTo({
      url: './focus/focus',
    })
  },

  // 跳转到签到页面
  navigateToCheckIn: function () {
    wx.navigateTo({
      url: './checkin/checkin',
    })
  }
})
