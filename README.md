# face
基于网络获取到[256,256,3] 的特征图，其中部分像素位置(预先定义好的)存储了顶点了x,y,z的信息，该x,y截断到[0,255]后取整就为u，v坐标，基于该uv坐标在原始的人脸rgb图中可以取出每个顶点对应的光照，来得到obj文件，[256,256,3]的x,y,z替换为对应的颜色后得到的[256,256,3]为texture图。



从人脸的三维模型到uv图的展开有专门的研究

