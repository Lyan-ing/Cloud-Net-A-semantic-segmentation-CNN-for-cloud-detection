import csv
import json
import os
import numpy as np
from osgeo import gdal, ogr, osr
import cv2
from loguru import logger
import traceback
import argparse
import tqdm

# gdal.SetConfigOption('CPL_LOG', 'OFF')
np.seterr(divide='ignore', invalid='ignore')

logger.add('logger/cloud_detection_error.logs', level='ERROR')
cloud_seg_mask_for_different_sensor = {}
if os.path.exists('cloud_seg_mask_for_different_sensor.json'):
    with open('cloud_seg_mask_for_different_sensor.json', 'r', encoding='utf-8') as js:
        cloud_seg_mask_for_different_sensor = json.load(js)

def get_alpha(input_image):
    ds = gdal.Open(input_image)
    bands_count = ds.RasterCount
    for i in range(1, bands_count + 1):
        band = ds.GetRasterBand(i)
        logger.info(f'{i}--{band.GetDescription()}')
        logger.info(f'maskFlags:{band.GetMaskFlags()}')
        logger.info(band.GetMaskBand() is None)

    band_mask = ds.GetRasterBand(5)


def detect_clouds(input_image, shapefilename):
    try:
        # get_alpha(input_image)
        # return
        # 打开遥感影像
        ds_vrt = gdal.Warp('', input_image, format='vrt')
        sensor_type = os.path.basename(input_image).split('_')[0]
        res = 0.0001
        options = gdal.BuildVRTOptions(xRes=res, yRes=res, resampleAlg=gdal.GRA_Average)
        dataset = gdal.BuildVRT('', ds_vrt)
        # dataset = gdal.Open(input_image)

        if dataset is None:
            raise Exception('无法打开影像文件')

        # 获取影像的基本信息
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        bands = dataset.RasterCount
        if bands < 4:
            raise Exception(f"波段数不足4")
        geotransform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()

        bandred = dataset.GetRasterBand(3)

        data_blue = dataset.GetRasterBand(1).ReadAsArray()
        data_green = dataset.GetRasterBand(2).ReadAsArray()
        data_red = dataset.GetRasterBand(3).ReadAsArray()
        data_nir = dataset.GetRasterBand(4).ReadAsArray()

        mask_nodata = None

        # mask_band = bandred.GetMaskBand()
        # if mask_band is not None:
        #     mask_data = mask_band.ReadAsArray()
        #     mask_nodata = mask_data==0
        # else:
        if 1:
            nodata = bandred.GetNoDataValue()
            if nodata is not None:
                mask_nodata = data_red == nodata
            else:
                mask_nodata = data_red == 0

        # rgb计算灰度
        # data_gray = 0.2989 * data_red + 0.5870 * data_green + 0.1140 * data_blue

        # data_gray = cv2.GaussianBlur(data_gray, (3, 3), 0)
        data_red = cv2.GaussianBlur(data_red, (5, 5), 0)
        data_nir = cv2.GaussianBlur(data_nir, (5, 5), 0)

        data_nir_nor = (data_nir - data_nir.min()) / (data_nir.max() - data_nir.min())
        # data_red_nor = (data_red - data_red.min()) / (data_red.max() - data_red.min())

        _cloud_mask_data = (data_nir - data_red) / (data_red + data_nir).clip(1)
        # _cloud_mask_data = (data_nir_nor - data_red_nor) / (data_red_nor + data_nir_nor)
        # cloud_mask_data = (data_gray-data_red)/(data_gray+data_nir)

        cloud_mask_data = np.zeros_like(_cloud_mask_data)
        # _cloud_mask_data = (_cloud_mask_data - _cloud_mask_data.min()) / (
        #             _cloud_mask_data.max() - _cloud_mask_data.min())
        # print(_cloud_mask_data.max())
        choose_threshold = False
        if sensor_type not in cloud_seg_mask_for_different_sensor:
            choose_threshold = True
        else:
            sensor_type_value = cloud_seg_mask_for_different_sensor[sensor_type]
            is_cloud_value = sensor_type_value['threshold']
            nir_value = sensor_type_value['nir_value']
        if choose_threshold:
            def nothing(x):
                pass

            cv2.namedWindow('Image')
            cv2.namedWindow('mask')
            cv2.resizeWindow('mask', 600, 600)

            # 创建一个滑动条，名为 'R'，范围为 0-255，初始值为 100
            cv2.createTrackbar('threshold', 'mask', 0, 100, nothing)
            cv2.createTrackbar('nir', 'mask', 0, 100, nothing)
            Print_flag = True
            print()
            while (1):
                k = cv2.waitKey(1) & 0xFF
                # 将波段数据合并成一个图像
                rgb = np.dstack((data_blue, data_green, data_red))

                # 将 NumPy 数组转换为 OpenCV 图像
                rgb_cv = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                resize_img = cv2.resize(rgb_cv, (600, 600))


                # 在 OpenCV 窗口中显示图像
                cv2.imshow('Image', resize_img)

                is_cloud_value = cv2.getTrackbarPos('threshold', 'mask') / 10
                nir_value = cv2.getTrackbarPos('nir', 'mask') / 100
                cloud_mask = (_cloud_mask_data > is_cloud_value) & (data_nir_nor > nir_value)

                cloud_mask_data[cloud_mask] = 1
                cloud_mask_data[~cloud_mask] = 0
                if Print_flag:
                    logger.info("按Q 确认选值 按A查看噪点优化后的结果")
                    Print_flag = False
                if k == ord('a'):
                    MORPH_ELLIPSE_FLAG = True
                else:
                    MORPH_ELLIPSE_FLAG = False
                if MORPH_ELLIPSE_FLAG:
                    logger.info("=============噪点优化中，请等候================")
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))  # 定义矩形结构元素

                    # cloud_mask_data = cv2.morphologyEx(cloud_mask_data, cv2.MORPH_OPEN, kernel,iterations=1)
                    cloud_mask_data = cv2.morphologyEx(cloud_mask_data, cv2.MORPH_CLOSE, kernel, iterations=1)
                    # cloud_mask_data = cv2.morphologyEx(cloud_mask_data, cv2.MORPH_GRADIENT, kernel,iterations=1)

                    cloud_mask_data = cv2.erode(cloud_mask_data, kernel, iterations=3)
                    # cloud_mask_data = cv2.dilate(cloud_mask_data, kernel, iterations=1)
                    cloud_mask_data[cloud_mask_data <= 0.5] = 0
                    cloud_mask_data[cloud_mask_data > 0.5] = 1
                    mask = cv2.resize(cloud_mask_data, (600, 600))

                    cv2.imshow('mask', mask)
                    # print()
                    logger.info('按C重新选择阈值或确认阈值')
                    while True:
                        kk = cv2.waitKey(1) & 0xFF
                        # print('按C重新选择阈值')
                        if kk == ord('c'):
                            Print_flag = True
                            break

                else:
                    mask = cv2.resize(cloud_mask_data, (600, 600))

                    cv2.imshow('mask', mask)
                # if cv2.getTrackbarPos(switch, 'mask') == 1:
                #     break
                if k == 27 or k==ord('q'):
                    break
                # 获取滑动条的位置

            # return None
            cv2.destroyAllWindows()
            if sensor_type not in cloud_seg_mask_for_different_sensor:
                cloud_seg_mask_for_different_sensor[sensor_type] = {'nir_value': nir_value, 'threshold': is_cloud_value}
        else:
            cloud_mask = (_cloud_mask_data > is_cloud_value) & (data_nir_nor > nir_value)
            # 计算云占比
            cloud_mask_data[cloud_mask] = 1
            cloud_mask_data[~cloud_mask] = 0
            # 使用闭运算优化噪点
            logger.info("=============噪点优化中，请等候================")
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))  # 定义矩形结构元素

            # cloud_mask_data = cv2.morphologyEx(cloud_mask_data, cv2.MORPH_OPEN, kernel,iterations=1)
            cloud_mask_data = cv2.morphologyEx(cloud_mask_data, cv2.MORPH_CLOSE, kernel, iterations=1)
            # cloud_mask_data = cv2.morphologyEx(cloud_mask_data, cv2.MORPH_GRADIENT, kernel,iterations=1)

            cloud_mask_data = cv2.erode(cloud_mask_data, kernel, iterations=3)
            # cloud_mask_data = cv2.dilate(cloud_mask_data, kernel, iterations=1)
            cloud_mask_data[cloud_mask_data <= 0.5] = 0
            cloud_mask_data[cloud_mask_data > 0.5] = 1
        # cloud_mask_data[data_nir<200] = 1
        # cloud_mask_data[data_nir<1000]=1
        # cloud_mask_data[data_gray<1000]=1

        # 计算云占比
        cloud_rate = round((cloud_mask.sum() / (width * height)) * 100, 2)


        if 1:
            # 关闭数据集
            dataset = None

            output_mask = input_image[:-5] + '_mask.tiff'
            # 创建云掩膜栅格数据
            driver = gdal.GetDriverByName("GTiff")
            cloud_mask = driver.Create(output_mask, width, height, 1, gdal.GDT_Float32)
            cloud_mask.SetGeoTransform(geotransform)
            cloud_mask.SetProjection(projection)
            cloud_mask.GetRasterBand(1).WriteArray(_cloud_mask_data)
            cloud_mask.GetRasterBand(1).SetNoDataValue(1)
            cloud_mask = None

        # 检测云并生成云掩膜
        # blue_band = data[:, :, 0]
        # green_band = data[:, :, 1]
        # cloud_threshold = 200  # 设置云阈值，根据实际情况调整
        # cloud_mask_data = np.where((blue_band > cloud_threshold) & (green_band > cloud_threshold), 1, 0).astype(np.uint8)
        # _cloud_mask_data = (data_nir-data_red)/(data_red+data_nir)

        # # 归一化 data_nir

        # _cloud_mask_data = (data_nir_nor-data_red_nor)/(data_red_nor+data_nir_nor)
        # _cloud_mask_data[_cloud_mask_data>1]=0

        # 关闭数据集
        # dataset = None
        # cloud_mask = None

        data_bounds = np.where(mask_nodata == 0, 0, 1)
        polygon_bounds = getVector(data_bounds)

        polygon_list = getVector(cloud_mask_data)
        # shapefilename = mask[:-4]+'.shp'
        write_shapefile(polygon_bounds, polygon_list, shapefilename, geotransform, projection)

        with open(os.path.join(args.tif_path, 'cloud_rate.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([input_image, cloud_rate])
        return True
    except Exception as ex:
        logger.error(f'云量检测失败：{input_image}')
        # logger.error(ex.args)
        logger.error(traceback.format_exc())
        return False


def write_shapefile(polygon_bounds, polygon_list, shapefile, geotransform, projection):
    """
    将多边形数据写入shape文件
    :param polygon_list: 多边形数据
    :param shapefile: shape文件名
    :return:
    """
    driver = ogr.GetDriverByName('ESRI Shapefile')
    datasource = driver.CreateDataSource(shapefile)
    srs = osr.SpatialReference()
    if projection:
        srs.ImportFromWkt(projection)
    else:
        srs.ImportFromEPSG(4326)

    layer: ogr.Layer = datasource.CreateLayer(os.path.splitext(shapefile)[0], srs=srs, geom_type=ogr.wkbPolygon)
    layer.CreateField(ogr.FieldDefn('type', ogr.OFTInteger))

    tran = np.array(geotransform).reshape((2, 3))

    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetField('type', 0)
    _poly = np.dot(polygon_bounds, tran[:, 1:]) + tran[:, 0]
    # _poly格式化成wkt
    wkt = [f'{x} {y}' for x, y in _poly[0, :, 0, :]]
    wkt.append(wkt[0])

    wkt = ','.join(wkt)
    wkt = f'POLYGON(({wkt}))'

    geom_bounds: ogr.Geometry = ogr.CreateGeometryFromWkt(wkt)
    # logger.info(geom_bounds.IsValid())

    feature.SetGeometry(geom_bounds)
    feature.SetField('type', 0)
    layer.CreateFeature(feature)

    for polygon in polygon_list:
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField('type', 1)
        _poly = np.dot(polygon, tran[:, 1:]) + tran[:, 0]
        # _poly格式化成wkt
        wkt = [f'{x} {y}' for x, y in _poly[:, 0, :]]
        wkt.append(wkt[0])

        # POINT_wkt = [f'POINT({x} {y})' for x,y in _poly[:,0,:]]
        # POINT_wkt = '\n'.join(POINT_wkt)
        wkt = ','.join(wkt)
        wkt = f'POLYGON(({wkt}))'
        _poly = ogr.CreateGeometryFromWkt(wkt)
        _poly = _poly.MakeValid()
        # logger.info(_poly.IsValid())
        geom = geom_bounds.Intersection(ogr.CreateGeometryFromWkt(wkt))
        if geom is None or geom.IsEmpty():
            continue

        # logger.info(geom.GetGeometryType())

        feature.SetGeometry(geom)
        layer.CreateFeature(feature)

    datasource.Destroy()


def getVector(input_data):
    # logger.info("getVector")
    data = np.zeros(input_data.shape, dtype='uint8')
    data[input_data == 0] = 255
    # contours, hierarchy = cv2.findContours(data, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # logger.info(contours)
    part_conts, hierarchy2 = cv2.findContours(data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    polygon_list = []
    area_max = 0
    polygon_Max = None
    epsilon = 1

    for i, part_cont in enumerate(part_conts):
        _area = cv2.contourArea(part_cont)
        # logger.info(_area)

        if _area > 10:
            polygon_Max = cv2.approxPolyDP(part_cont, epsilon, True)
            polygon_list.append(polygon_Max)
    return polygon_list


if __name__ == '__main__':
    logger.info('开始进行云量检测')
    tif_file_list = []
    parser = argparse.ArgumentParser(description="生成tif文件云数据")

    # 添加参数
    parser.add_argument("tif_path", help="tif文件目录")

    # 解析命令行参数
    args = parser.parse_args()

    for root, dirs, files in os.walk(args.tif_path):

        # tif_path = r'E:\云量测试数据'
        # for root,dirs,files in os.walk(tif_path):
        for tif in files:
            if not os.path.splitext(tif)[-1].lower() in ['.tiff', '.tif']:
                continue
            _tif = os.path.join(root, tif)
            tif_file_list.append(_tif)

    erro_list = []
    for tif in tqdm.tqdm(tif_file_list):
        shp = '.'.join(tif.split('.')[:-1]) + '_cloud.shp'
        if not detect_clouds(tif, shp):
            erro_list.append(tif)
    with open('cloud_seg_mask_for_different_sensor.json', 'w', encoding='utf-8') as clo:
        json.dump(cloud_seg_mask_for_different_sensor, clo, indent=4)
    if erro_list:
        logger.info(f'云量检测计算完成,有{len(erro_list)}个文件生成失败，请检测日志文件')
    else:
        logger.info("云量检测计算完成")

    # output_folod = 'mask16'

    # if not os.path.exists(output_folod):
    #     os.makedirs(output_folod)
    # for tif in os.listdir('云量检测'):
    #     if not tif.endswith('.tif'):
    #         continue
    #     _tif = os.path.join('云量检测',tif)
    #     mask = os.path.join(output_folod, tif)
    #     detect_clouds(_tif, mask)

# 示例用法
# input_image = r"云量检测\oSV1-04_20220713_L1B0001352428_6042300610380038_01-MUX.tif"
# output_vector = r"云量检测\oSV1-04_20220713_L1B0001352428_6042300610380038_01-MUX._mask7.tif"
# detect_clouds(input_image, output_vector)
