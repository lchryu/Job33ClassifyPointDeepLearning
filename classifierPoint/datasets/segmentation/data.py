import gvlib
import laspy
import numpy as np
from pathlib import Path
from typing import Union, List, Optional
import shutil

AVIABLEEXT = ["las", "LiData", "laz"]


def recursive_split(x_min, y_min, x_max, y_max, max_x_size, max_y_size):
    x_size = x_max - x_min
    y_size = y_max - y_min
    if x_size > max_x_size:
        left = recursive_split(x_min, y_min, x_min + (x_size // 2), y_max, max_x_size, max_y_size)
        right = recursive_split(x_min + (x_size // 2), y_min, x_max, y_max, max_x_size, max_y_size)
        return left + right
    elif y_size > max_y_size:
        up = recursive_split(x_min, y_min, x_max, y_min + (y_size // 2), max_x_size, max_y_size)
        down = recursive_split(x_min, y_min + (y_size // 2), x_max, y_max, max_x_size, max_y_size)
        return up + down
    else:
        return [(x_min, y_min, x_max, y_max)]


class LidataIterator:
    def __init__(self, reader: gvlib.LidataReader, sub_bounds) -> None:
        self.reader = reader
        self.sub_bounds = sub_bounds
        self.ptr = 0
        self.end = len(self.sub_bounds)

    def __next__(self):
        if self.ptr < self.end:
            self.reader.read_boxf(self.sub_bounds[self.ptr])
            lidata = self.reader.tile()
            pos = lidata.xyz
            classification = lidata.classification
            rgb = lidata.rgb
            intensity = lidata.intensity
            self.ptr += 1
            return pos, classification, rgb, intensity
        else:
            raise StopIteration

    def __iter__(self) -> "LidataIterator":
        return self


class LasIterator:
    def __init__(self, las_path: Path) -> None:
        self.las_path = las_path
        self.las_files = list(self.las_path.glob("*.las"))
        self.ptr = 0
        self.end = len(self.las_files)

    def __next__(self):
        if self.ptr < self.end:
            las = laspy.read(self.las_files[self.ptr])
            pos = las.xyz
            classification = las.classification
            if hasattr(las, "red"):
                rgb = np.c_[las.red, las.green, las.blue]
            else:
                rgb = np.zeros(pos.shape, dtype=np.float32)
            if hasattr(las, "intensity"):
                intensity = las.intensity
            else:
                intensity = np.zeros(classification.shape, dtype=np.int32)
            self.ptr += 1
            return pos, classification, rgb, intensity
        else:
            shutil.rmtree(self.las_path, ignore_errors=True)
            raise StopIteration

    def __iter__(self) -> "LasIterator":
        return self


class LasDirectIterator:
    def __init__(self, file_path: Path, sub_bounds) -> None:
        self.file_path = file_path
        self.sub_bounds = sub_bounds
        self.ptr = 0
        self.end = len(sub_bounds)
        
    def __next__(self):
        if self.ptr < self.end:
            # Đọc dữ liệu trong phạm vi bounds hiện tại
            x_min, y_min, z_min, x_max, y_max, z_max = self.sub_bounds[self.ptr]
            
            # Đọc file LAS và lọc điểm trong phạm vi
            with laspy.open(self.file_path) as file:
                point_data = []
                # Đọc theo từng chunk để tránh tải toàn bộ file vào bộ nhớ
                for points in file.chunk_iterator(1000000):
                    # Lọc điểm trong phạm vi
                    mask = ((points.x >= x_min) & (points.x <= x_max) & 
                           (points.y >= y_min) & (points.y <= y_max) &
                           (points.z >= z_min) & (points.z <= z_max))
                    
                    if np.any(mask):
                        point_data.append(points[mask])
                
                if not point_data:
                    self.ptr += 1
                    return np.zeros((0, 3)), np.zeros(0), np.zeros((0, 3)), np.zeros(0)
                
                # Gộp các mảng điểm lại
                if len(point_data) > 1:
                    # Alternative to laspy.merge_points()
                    # Extract all the needed data directly and concatenate the numpy arrays
                    
                    # Initialize arrays for concatenation
                    all_x = []
                    all_y = []
                    all_z = []
                    all_classification = []
                    all_red = []
                    all_green = []
                    all_blue = []
                    all_intensity = []
                    has_rgb = False
                    has_intensity = False
                    
                    # Extract data from each point cloud
                    for points in point_data:
                        all_x.append(points.x)
                        all_y.append(points.y)
                        all_z.append(points.z)
                        all_classification.append(points.classification)
                        
                        if hasattr(points, 'red'):
                            has_rgb = True
                            all_red.append(points.red)
                            all_green.append(points.green)
                            all_blue.append(points.blue)
                        
                        if hasattr(points, 'intensity'):
                            has_intensity = True
                            all_intensity.append(points.intensity)
                    
                    # Create final arrays
                    x = np.concatenate(all_x)
                    y = np.concatenate(all_y)
                    z = np.concatenate(all_z)
                    classification = np.concatenate(all_classification)
                    
                    # Create coordinates array
                    pos = np.vstack((x, y, z)).T
                    
                    # Create RGB array if available
                    if has_rgb:
                        rgb = np.vstack((
                            np.concatenate(all_red),
                            np.concatenate(all_green),
                            np.concatenate(all_blue)
                        )).T
                    else:
                        rgb = np.zeros(pos.shape, dtype=np.float32)
                    
                    # Create intensity array if available
                    if has_intensity:
                        intensity = np.concatenate(all_intensity)
                    else:
                        intensity = np.zeros(classification.shape, dtype=np.int32)
                    
                    # Return directly without using points object
                    self.ptr += 1
                    return pos, classification, rgb, intensity
                else:
                    # Single point cloud case
                    points = point_data[0]
                    
                    pos = np.vstack((points.x, points.y, points.z)).T
                    classification = points.classification
                    
                    if hasattr(points, 'red'):
                        rgb = np.vstack((points.red, points.green, points.blue)).T
                    else:
                        rgb = np.zeros(pos.shape, dtype=np.float32)
                    
                    if hasattr(points, 'intensity'):
                        intensity = points.intensity
                    else:
                        intensity = np.zeros(classification.shape, dtype=np.int32)
                    
                    self.ptr += 1
                    return pos, classification, rgb, intensity
        else:
            raise StopIteration
    
    def __iter__(self) -> "LasDirectIterator":
        return self


class DataReader:
    """return data iterator, may return empty data"""

    def __init__(self, scan_file: Path, max_point=2e7):
        self.data_reader = None
        self.ext = scan_file.suffix[1:]  # .lidata->lidata
        self.file_path = scan_file
        self._sub_bounds = []
        self.sub_point_cloud = 0
        self.max_point = max_point
        if self.ext == "LiData":
            self.data_reader = gvlib.LidataReader(str(scan_file))
            if not self.data_reader.open():
                raise IOError(f"can't open data {str(scan_file)}")
            # 此版本 box范围写反
            self.point_count = self.data_reader.point_count()
            self.box = self.data_reader.bbx_2f4()[[1, 0], :]
        elif self.ext == "las" or self.ext == "laz":
            with laspy.open(scan_file) as file:
                self.point_count = file.header.point_count
                self.box = np.array(
                    [
                        [file.header.x_min, file.header.y_min, file.header.z_min],
                        [file.header.x_max, file.header.y_max, file.header.z_max],
                    ]
                )
        else:
            raise IOError(f"don't support this type :{self.ext}")
        self._block_size = (self.box[1, :2] - self.box[0, :2]) / (np.sqrt(self.point_count / self.max_point))

    def chunk_iterator(self):
        if self.ext.lower() in ["las", "laz"]:
            return LasDirectIterator(self.file_path, self._sub_bounds)
        if self.ext.lower() == "lidata":
            return LidataIterator(self.data_reader, self._sub_bounds)

    def sub_bounds(self, block_size=None):
        if block_size is None:
            block_size = self._block_size
        if not self._sub_bounds:
            self.recursive_split(block_size=block_size)
        return self._sub_bounds


    def recursive_split(self, block_size=None):
        if block_size is None:
            block_size = self._block_size
        x_min = self.box[0, 0]
        x_max = self.box[1, 0]
        y_min = self.box[0, 1]
        y_max = self.box[1, 1]
        z_min = self.box[0, 2]
        z_max = self.box[1, 2]
        sub_bounds = recursive_split(x_min, y_min, x_max, y_max, block_size[0], block_size[1])
        self._sub_bounds = [
            list([x_min, y_min, z_min, x_max, y_max, z_max]) for (x_min, y_min, x_max, y_max) in sub_bounds
        ]

    def writer_boxf_lidata(self, block_size, classification):
        self.data_reader.read_boxf(block_size)
        lidata = self.data_reader.tile()
        lidata.classification = classification
        self.sub_point_cloud += classification.shape[0]
        lidata.dump()

    def read_boxf_lidata(self, block_size):
        """Đọc dữ liệu từ file LiData hoặc LAS trong phạm vi block_size"""
        if self.ext.lower() == 'lidata':
            # Đọc dữ liệu từ file LiData
            self.data_reader.read_boxf(block_size)
            lidata = self.data_reader.tile()
            pos = lidata.xyz
            rgb = lidata.rgb
            classification = lidata.classification
            intensity = lidata.intensity
            self.sub_point_cloud += classification.shape[0]
            return pos, classification, rgb, intensity
        elif self.ext.lower() in ['las', 'laz']:
            # Đọc dữ liệu từ file LAS/LAZ
            return self.read_boxf_las(block_size)
        else:
            raise ValueError(f"Không hỗ trợ định dạng file: {self.ext}")

    def read_boxf_las(self, block_size):
        """Đọc dữ liệu từ file LAS trong phạm vi block_size"""
        x_min, y_min, z_min, x_max, y_max, z_max = block_size
        
        # Đọc file LAS
        las = laspy.read(self.file_path)
        
        # Lọc điểm trong phạm vi block_size
        mask = (las.x >= x_min) & (las.x <= x_max) & \
               (las.y >= y_min) & (las.y <= y_max) & \
               (las.z >= z_min) & (las.z <= z_max)
        
        # Lấy dữ liệu điểm trong phạm vi
        pos = np.vstack((las.x[mask], las.y[mask], las.z[mask])).T
        
        # Lấy dữ liệu phân loại
        classification = las.classification[mask]
        
        # Lấy dữ liệu màu sắc
        if hasattr(las, 'red'):
            rgb = np.vstack((las.red[mask], las.green[mask], las.blue[mask])).T
        else:
            rgb = np.zeros(pos.shape, dtype=np.float32)
        
        # Lấy dữ liệu cường độ
        if hasattr(las, 'intensity'):
            intensity = las.intensity[mask]
        else:
            intensity = np.zeros(classification.shape, dtype=np.int32)
        
        self.sub_point_cloud += classification.shape[0]
        
        return pos, classification, rgb, intensity
