# this script requires Open3D python binding
import os
import numpy as np
import open3d as o3d


def read_dmap(file_name, order='little'):
    with open(file_name, mode='rb') as file:
        data = file.read()
    name = int.from_bytes(data[:2], signed=False, byteorder=order)
    mode = int.from_bytes(data[2:3], signed=False, byteorder=order)
    padding = int.from_bytes(data[3:4], signed=False, byteorder=order)
    imageWidth = int.from_bytes(data[4:8], signed=False, byteorder=order)
    imageHeight = int.from_bytes(data[8:12], signed=False, byteorder=order)
    depthWidth = int.from_bytes(data[12:16], signed=False, byteorder=order)
    depthHeight = int.from_bytes(data[16:20], signed=False, byteorder=order)
    data_array = np.frombuffer(data[20:28], dtype=np.float32)
    dMin = float(data_array[0])
    dMax = float(data_array[1])
    imageFileNameSize = int.from_bytes(data[28:30], signed=False, byteorder=order)
    i = 30+imageFileNameSize
    imageFileName = str(data[30:i])
    IDsSize = int.from_bytes(data[i:i+4], signed=False, byteorder=order)
    j = i+4+IDsSize*4
    neighbors = np.frombuffer(data[i+4:j], dtype=np.uint32)
    i = j
    j = i+9*8
    K = np.frombuffer(data[i:j], dtype=np.float64).reshape((3,3))
    invK = np.linalg.inv(K)
    i = j
    j = i+9*8
    R = np.frombuffer(data[i:j], dtype=np.float64).reshape((3,3))
    i = j
    j = i+3*8
    C = np.frombuffer(data[i:j], dtype=np.float64).reshape((3,1))
    [i,j] = [j,j+depthWidth*depthHeight*4]
    depthMap = np.frombuffer(data[i:j], dtype=np.float32).reshape((depthHeight,depthWidth))
    if mode&2 != 0:
        [i,j] = [j,j+depthWidth*depthHeight*3*4]
        normalMap = np.frombuffer(data[i:j], dtype=np.float32).reshape((depthHeight,depthWidth,3))
        xyz = np.empty((3,depthWidth*depthHeight), dtype=np.float64)
    xyz = np.empty((3,depthWidth*depthHeight), dtype=np.float64)
    nxyz = np.empty((3,depthWidth*depthHeight), dtype=np.float64)
    i = 0
    for r in range(depthHeight):
        for c in range(depthWidth):
            d = depthMap[r,c]
            if d > 0:
                xyz[:,i:i+1] = np.array([[float(c)],[float(r)],[1.0]])*d
                if mode&2 != 0:
                    nxyz[:,i:i+1] = normalMap[r,c,:].reshape(3,1)
                i = i+1
    xyz = xyz[:,:i]
    xyz = np.dot(invK,xyz)
    xyz = np.dot(R.T,xyz)+C
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.T)
    if mode&2 != 0:
        nxyz = nxyz[:,:i]
        nxyz = np.dot(R.T,nxyz)
        pcd.normals = o3d.utility.Vector3dVector(nxyz.T)
    else:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    return pcd


def read_pcd(file_name):
    if os.path.splitext(file_name)[1] == '.dmap':
        return read_dmap(file_name)
    print(file_name)
    return o3d.io.read_point_cloud(file_name)
