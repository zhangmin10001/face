"""
如何 从 Position Map 产生 3D Face
"""
import numpy as np
import os
from skimage.io import imread, imsave
import cv2
# 类似 超参数
uv_kpt_ind = np.loadtxt("datas/uv-data/uv_kpt_ind.txt").astype(np.int32)  # 2 x 68 get kpt 第一个为x,第二个为y 都是整数
face_ind = np.loadtxt("datas/uv-data/face_ind.txt").astype(np.int32)  # get valid vertices in the pos map 43867 都是整数，记录了每个顶点
triangles = np.loadtxt("datas/uv-data/triangles.txt").astype(np.int32)  # ntri x 3 86906 * 3 都是整数，三角面片的数目


# 测试一个例子
face_url = "datas/image00050.jpg"
face_texture_url = "datas/image00050_tex.jpg"
# Label 是 npy 数据， 可不是  position map图像(只是用来显示看的)
face_posmap_url = "datas/image00050.npy"


global resolution_op
resolution_op = 256
# 设置参数
uv_h = uv_w = 256
image_h = image_w = 256

image_posmap = np.load(face_posmap_url)
print(image_posmap.shape) # (256, 256, 3)
print(np.max(image_posmap)) #最大值249.7007

# save posmap
save_prefix = "results/"
pos_name = face_url.split("/")[-1].split(".")[0] + "_posmap.png"
pos_save_path = os.path.join(save_prefix, pos_name)
image_posmap_show = image_posmap.copy()
image_posmap_show = image_posmap_show.astype(np.uint8)
cv2.imwrite(pos_save_path,image_posmap_show)
cv2.imshow("pos_img",image_posmap_show)
cv2.waitKey(0)

# 从 Position Map 获取 顶点
def get_vertices(pos):
    '''
    Args:
        pos: the 3D position map. shape = (256, 256, 3).
    Returns:
        vertices: the vertices(point cloud). shape = (num of points, 3). n is about 40K here.
    '''
    all_vertices = np.reshape(pos, [resolution_op ** 2, -1])
    vertices = all_vertices[face_ind, :]  # face_ind 是什么呢？

    return vertices


def get_landmarks(pos):
    """
    Args:
        pos: the 3D position map. shape = (256, 256, 3).
    Returns:
        kpt: 68 3D landmarks. shape = (68, 3).
    :param pos:
    :return:
    """
    kpt = pos[uv_kpt_ind[1, :], uv_kpt_ind[0, :], :]
    return kpt


kpt = get_landmarks(image_posmap)
print(kpt.shape)  # (68, 3)  68个关键点
# face_id是提前约定好的posmap图中的哪些位置对应了人脸的3D点，这样就直接得到了ply的顶点信息
vertices = get_vertices(image_posmap) # image_posmap记录了3D人脸的x,y,z的位置，将256*256*3 reshape 为(256*256)*3的点云,并取出预先定义的face_id对应的位置，得到43867个点云
print(vertices.shape)  # (43867, 3) PRNet 人脸是 43867 个顶点


# 保存顶点  不保存纹理
def dump_to_ply(vertex, tri, wfp):
    header = """ply
    format ascii 1.0
    element vertex {}
    property float x
    property float y
    property float z
    element face {}
    property list uchar int vertex_indices
    end_header"""
    n_vertex = vertex.shape[1]  # ((3, 43867))
    n_face = tri.shape[1]   # ((3, 86906))
    header = header.format(n_vertex, n_face)

    with open(wfp, 'w') as f:
        f.write(header + '\n')
        for i in range(n_vertex):  # 顶点
            x, y, z = vertex[:, i]
            f.write('{:.4f} {:.4f} {:.4f}\n'.format(x, y, z))
        for i in range(n_face):  # 三角形
            idx1, idx2, idx3 = tri[:, i]
            f.write('3 {} {} {}\n'.format(idx1 - 1, idx2 - 1, idx3 - 1))
    print('Dump tp {}'.format(wfp))



name = face_url.split("/")[-1].split(".")[0] + ".ply"
print(name)
face_ply = os.path.join(save_prefix, name)

# 保存 顶点信息 shape 成功
dump_to_ply(vertices.T, triangles.T, face_ply)   # 切记 tri 是 /Data/uv-data/triangles.txt 中的三角


# 保存 带上 color/texture 信息
def get_colors(image, vertices): #本质上直接取三维x,y,z中的x,y的位置，只是对其进行了[0,w-1]和[0,h-1]的截断以及取整操作，基于x,y坐标在原图上获取颜色
    """
        Args:
            pos: the 3D position map. shape = (256, 256, 3).
        Returns:
            colors: the corresponding colors of vertices. shape = (num of points, 3). n is 45128 here.
    """
    [h, w, _] = image.shape
    vertices[:, 0] = np.minimum(np.maximum(vertices[:, 0], 0), w - 1)  # x
    vertices[:, 1] = np.minimum(np.maximum(vertices[:, 1], 0), h - 1)  # y
    ind = np.round(vertices).astype(np.int32)
    colors = image[ind[:, 1], ind[:, 0], :]  # n x 3

    return colors

def get_uv_map(image_posmap,image_face):
    """
        Args:
            image : ori color image
            vertices : x,y,z
        Returns:
            image_tex: texture map
    """
    [h, w, _] = image_posmap.shape
    image_tex = image_face.copy()
    u_pixel = np.round(np.clip(image_posmap[:,:,0],0,w-1)).astype(np.int32)
    v_pixel = np.round(np.clip(image_posmap[:,:,1],0,h-1)).astype(np.int32)
    image_tex[:,:,:] = image_face[v_pixel,u_pixel]
    return image_tex

image_face = imread(face_url)  # face_url 是 剪切后为（256， 256， 3）的人脸图像
[h, w, c] = image_face.shape
print(h, w, c)
image_face = image_face / 255.
colors = get_colors(image_face, vertices)  # 从人脸 和 顶点 中获取 color (43867, 3)
print(colors.shape)


# 写入 .obj文件，具有colors （texture）
def write_obj_with_colors(obj_name, vertices, triangles, colors):
    ''' Save 3D face model with texture represented by colors.
    Args:
        obj_name: str
        vertices: shape = (nver, 3)
        colors: shape = (nver, 3)
        triangles: shape = (ntri, 3)
    '''
    triangles = triangles.copy()
    triangles += 1  # meshlab start with 1

    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'

    # write obj
    with open(obj_name, 'w') as f:

        # write vertices & colors
        for i in range(vertices.shape[0]):
            # s = 'v {} {} {} \n'.format(vertices[0,i], vertices[1,i], vertices[2,i])
            s = 'v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0],
                                               colors[i, 1], colors[i, 2])
            f.write(s)

        # write f: ver ind/ uv ind
        [k, ntri] = triangles.shape
        for i in range(triangles.shape[0]):
            # s = 'f {} {} {}\n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2])
            s = 'f {} {} {}\n'.format(triangles[i, 2], triangles[i, 1], triangles[i, 0])
            f.write(s)
name = face_url.split("/")[-1].split(".")[0] + ".obj"
save_vertices = vertices.copy()
save_vertices[:, 1] = h - 1 - save_vertices[:, 1]  # 这一步 不可缺少； (43867, 3)
write_obj_with_colors(os.path.join(save_prefix, name), save_vertices, triangles, colors)  # save 3d face(can open with meshlab)

image_texture = get_uv_map(image_posmap,image_face)
image_texture = (image_texture * 255).astype(np.uint8) # 将其恢复为255，并且转化为uint8格式，否则在cvtColor中会报错
# image_texture = image_texture.transpose(2,1,0)
image_texture = cv2.cvtColor(image_texture, cv2.COLOR_BGR2RGB)
print("texture.shape : {}".format(image_texture.shape))
cv2.imshow("texture",image_texture)
pic_name = face_url.split("/")[-1].split(".")[0] + ".png"
pic_save_path = os.path.join(os.path.join(save_prefix, pic_name))
cv2.imwrite(pic_save_path,image_texture)
cv2.waitKey(0)
