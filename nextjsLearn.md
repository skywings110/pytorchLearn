# 1. 服务器组件和浏览器组件
默认server，在next的后端执行，是一种完成的html代码，发给前端，js代码少，引擎优化比较好，web爬虫可以看到完整页面
use client才会在浏览器执行，如果用客户端代码，那么html是空的，全是js实现的，用户交互就需要客户端组件
尽量保留server组件，性能较好，只有必须用client才用client
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
## 2.4. 可以进行外包组件
在app/下面可以新建component/然后放里面页面需要的component
## 2.5. 获取当期url路径名字符串
```jsx
import { usePathname } from 'next/navigation' 
// path就是url路径
const path = usePathname()
```
## 2.6. 动态段路由值
`${}`
```python
# 图片
import logoImg from "./../logo.png"
# 这种情况下next可以直接查找高度和宽度
src = {logoImg}
# src={路径}时必须有宽和高，所以加fill意味着直接填充，不知道的图直接fill是比较好的处理方案。
```
提取所有属性用...比如<MealItem {...meal}/>
meal是一个键值对
function MealItem({title, slug, image, summary})这种就可以自动匹配上
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
### 3.2.2


# 4. page and layout
## 4.1. layout
layout的东西可以改变下面的所有page，通常做最上面的导航栏比较好

## 4.2. 增加loading page

# 5. 组件
## 5.1. Image
默认loading = "lazy"，格式是webp，比png更有效，像页面logo这种没有内容转移或者闪烁的就优先加载。这种在Image 标签里加priority

## 5.2. 换行
可以用replace
比如，meal.instructions = meal.instructions.replace(/\n/g, '<br/>');

## 5.3. 直接输出段落
可以在<div>中的属性写dangerouslySetInnerHTML={{__html: '...'}}，__html是html字符串
 
# 6. 动画
## 6.1. setInterval
做图片间隔变化，使用useEffect

# 7. js基础
## 7.1. startWith
js判断是否以某个字符串开头，判断状态
```jsx
<Link href="/meals" className={path.startsWith('/meals') ? classes.active : undefined}>
```

# 8. SQLite Database
略

# 9. fetch data
# 9.1. 原生用法
```jsx
// 原生用法
useEffect(() => {
  fetch()
}, []);
```
# 9.2. server端调用
因为不写server端的查询，所以db.prepare(查询语句)这块略去
服务器组件可以转换为异步函数
在服务器组件里使用await 调取函数值，即const xx = await getxx()就可以直接用