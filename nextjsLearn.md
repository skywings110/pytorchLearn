# 1. 服务器组件和浏览器组件

# 2. 路由
## 2.1. 显示
app下面/about的page.jsx文件才算数  
除了page.jsx还有layout, not-found, error
## 2.2. 导航
### 2.2.1 a标签
缺点：每次点击都要重新加载
### 2.2.2 Link
import Link from next/link
Link标签不会找server端处理，直接就显示，实现客户端路由。  
客户端路由是指在浏览器中，通过JavaScript来管理和控制页面的加载和渲染，而不是依赖于服务器端的页面跳转。当用户点击Link组件时，实际上是在客户端触发了一个路由的变更事件，而不是向服务器发送请求来获取新的HTML页面。