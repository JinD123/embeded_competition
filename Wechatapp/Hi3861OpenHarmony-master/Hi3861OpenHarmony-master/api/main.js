import createRequest from '../utils/request'

export function loginRequest(data) {
  return createRequest({
    url: '/login',
    method: 'POST',
    data,
    needLogin: false
  })
}

const PageData = {
  data: {
    stuId: 'test', // 学号
    password: '1234', // 密码
    saveCount: true, // 是否记住账号，默认选中
  },

  onLoad(options) {
    this.initAccount()
  },

  initAccount() {
    const accountCache = wx.getStorageSync("account")
    if (accountCache) {
      this.setData({
        ...accountCache
      })
    }
  },

  login() {
    const that = this
    const postData = {
      stuId: that.data.stuId,
      password: that.data.password
    }
    wx.showLoading({
      title: '登录中',
    })
    loginRequest(postData).then(res => {
      wx.hideLoading()
      if (res.code == -1) {
        wx.showToast({
          title: res.msg,
          icon: 'none'
        })
        return
      }
      if (that.data.saveCount) {
        wx.setStorageSync('account', postData)
      }
      wx.setStorageSync('token', res.data.cookie)
      wx.showToast({
        title: '登录成功',
        icon: 'none'
      })
      setTimeout(() => {
        wx.redirectTo({
          url: '/pages/index/index',
        })
      }, 1500);
    })
  },

  switchStatus() {
    this.setData({
      saveCount: !this.data.saveCount
    })
  }
}

Page(PageData)