# 1. 服务器组件和浏览器组件

# 2. 路由
## 2.1. 显示
app下面/about的page.jsx文件才算数  
除了page.jsx还有layout, not-found, error
## 2.2. 动态路由
可以使用[]
## 2.3. 导航
### 2.2.1 a标签
缺点：每次点击都要重新加载
### 2.2.2 Link
import Link from next/link
Link标签不会找server端处理，直接就显示，实现客户端路由。  
客户端路由是指在浏览器中，通过JavaScript来管理和控制页面的加载和渲染，而不是依赖于服务器端的页面跳转。当用户点击Link组件时，实际上是在客户端触发了一个路由的变更事件，而不是向服务器发送请求来获取新的HTML页面。
## 2.4.可以进行外包组件
在app/下面可以新建component/然后放里面页面需要的component

# 3. css
## 3.1. icon.jpg
更改网页上方的图标
## 3.2. css
建立一个*.module.css的文件，比如page.mdoule.css  
然后import classes from './page.module.css'  
然后下面需要这个格式的就写<div className={classes.logo}></div>
### 3.2.1 page.module.css里面就是
.logo {
  margin: 0;
  padding: 0;
}

# 4. page and layout
## 4.1. layout
layout的东西可以改变下面的所有page，通常做最上面的导航栏比较好

# 5. 组件
## 5.2. <Image>
默认loading = "lazy"，格式是webp，比png更有效，像页面logo这种没有内容转移或者闪烁的就优先加载。这种在<Image priority>
