import os
from pathlib import Path
import numpy as np
import multiprocessing
import logging
import torch
from torch_geometric.data import Dataset, Data
from typing import Dict, Any, Sequence, List
from classifierPoint.datasets.base_dataset import BaseDataset
from classifierPoint.metrics.segmentation_tracker import SegmentationTracker
from omegaconf import OmegaConf
from classifierPoint.utils.util import check_status, processbar, init_classmapping
from classifierPoint.utils.status_code import STATUS
from classifierPoint.utils.config_unpack import header_from_data
import shutil
from classifierPoint.datasets.segmentation.data import DataReader, AVIABLEEXT
import laspy

basic_log = logging.getLogger(__name__)
log = logging.getLogger("metrics")


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


def OmegaConf2str(conf: OmegaConf):
    if len(conf) == 0:
        dict_config = OmegaConf.to_container(conf)
    else:
        dict_config = OmegaConf.to_container(conf)[0]
    str_config = str(dict_config)
    return str_config


class LidarClassify(Dataset):
    def __init__(
        self,
        data_path: str,
        classmapping: Dict,
        transform=None,
        pre_transform=None,
        split_transform=None,
        pre_transform_dict={},
        split_transform_dict={},
        process_workers=1,
    ):
        self.classmapping = classmapping
        self.classes = []
        self.data_classesid = set()
        self.use_multiprocessing = process_workers > 1
        self.process_workers = process_workers
        self.pre_transform_dict = pre_transform_dict
        self.split_transform = split_transform
        self.split_transform_dict = split_transform_dict
        super().__init__(data_path, transform=transform, pre_transform=pre_transform)
        self.reversal_classmapping = None
        self._process()
        self._scans = list(self.data_dir.glob("*.pt"))
        self._num_features = self.num_node_features

    @property
    def data_dir(self) -> Path:
        return Path(os.path.join(self.root, "processed_data"))

    @property
    def conf_dir(self) -> Path:
        return self.data_dir / "conf"

    @property
    def conf_file(self) -> Path:
        return self.conf_dir / "pre_transform.pt"

    def check_pretransfrom(self, pre_transform: Dict, split_transform: Dict):
        if self.conf_file.is_file():
            tile_info = torch.load(self.conf_file)
            if tile_info["pre_transform"] == pre_transform and tile_info["split_transform"] == split_transform:
                return True
        logging.warning(
            f"The `pre_transform` argument differs from the one used in "
            f"the pre-processed version of this dataset. If you want to "
            f"make use of another pre-processing technique, make sure to "
            f"sure to delete '{self.data_dir}' first"
        )
        return False

    def process_one(self, scan_file: Path, pre_transform, split_transform):
        if check_status():
            return STATUS.PAUSE
        big_data = DataReader(scan_file)
        big_data.recursive_split()
        chunk_num = 0
        for xyz, cls, rgb, intensity in big_data.chunk_iterator():
            if not np.any(xyz) or xyz.shape[0] < 100:
                continue
            xyz = xyz - xyz.min(0)
            rgb = torch.tensor(rgb / 255, dtype=torch.float32)
            intensity = torch.tensor(intensity / 65535, dtype=torch.int32)
            data = Data(
                pos=torch.tensor(xyz, dtype=torch.float32),
                y=torch.tensor(cls, dtype=torch.long),
                rgb=rgb,
                intensity=intensity,
            )
            if pre_transform:
                data = pre_transform(data)
            if split_transform:
                data_dict = split_transform(data)
                data_list = data_dict["data"]
                index = 0
                for split in data_list:
                    out_file = self.data_dir / (scan_file.stem + "_" + str(chunk_num) + "_" + str(index) + ".pt")
                    index += 1
                    delattr(split, "splitidx")
                    torch.save(split, out_file)
            else:
                out_file = self.data_dir / (scan_file.stem + "_" + str(chunk_num) + ".pt")
                torch.save(data, out_file)
            classes = torch.unique(data.y, sorted=False).tolist()
            self.data_classesid.update(classes)
            chunk_num += 1
            log.info("Processed file %s, points number = %i", scan_file.stem + "_" + str(chunk_num), data.pos.shape[0])
        return STATUS.SUCCESS

    def _process(self):
        res = STATUS.SUCCESS
        if self.data_dir.is_dir() and self.conf_file.is_file():
            if not self.classmapping:
                data_classes = header_from_data(self.root, external_mode=False)["data_classesid"]
                # default classmap
                tmp_remap = {key: i + 1 for i, key in enumerate(data_classes)}
                self.classmapping, self.reversal_classmapping = init_classmapping(tmp_remap)
            self.classes = set(list(self.classmapping.values()))
            classes_num = max(self.classmapping.values()) + 1
            if classes_num > len(self.classes):
                raise Exception(f"data classes:{classes_num} is more than model classes {self.classes}")

            if self.check_pretransfrom(self.pre_transform_dict, self.split_transform_dict):
                log.info("Data has been processed,skip")
                return res
            else:
                log.info(f"will delete {self.data_dir}")
                try:
                    shutil.rmtree(self.conf_dir)
                    shutil.rmtree(self.data_dir)
                except OSError as e:
                    basic_log.error(f"{self.data_dir}:{e.strerror}")

        self.data_dir.mkdir(exist_ok=True)
        self.conf_dir.mkdir(exist_ok=True)
        scan_paths = []
        for ext in AVIABLEEXT:
            scan_paths = scan_paths + list(Path(self.root).glob(f"**/*.{ext}"))
        # scan_paths = scan_paths[:10]
        total_data = len(scan_paths)
        if not total_data >= 1:
            log.info(f"error !!!empty data,please check data dir")
            raise Exception(f"error !!!empty data,please check data dir")
        args = zip(
            scan_paths,
            [self.pre_transform for i in range(len(scan_paths))],
            [self.split_transform for i in range(len(scan_paths))],
        )

        if not self.use_multiprocessing:
            for index, arg in enumerate(args):
                processbar("Dataprepare", index, total_data)
                tmp = self.process_one(*arg)
                if tmp == STATUS.PAUSE:
                    res = STATUS.PAUSE
                    break
        elif self.use_multiprocessing:
            p = multiprocessing.Pool(self.process_workers)
            index = 0

            def quit(arg):
                nonlocal index
                nonlocal res
                index += 1
                processbar("Dataprepare", index, total_data)
                if arg == STATUS.PAUSE:
                    res = STATUS.PAUSE
                    p.terminate()  # kill all pool workers

            for i in range(self.process_workers):
                p.starmap_async(self.process_one, args, callback=quit)
            p.close()
            p.join()

        tile_info = {"pre_transform": OmegaConf.to_container(self.pre_transform_dict)}
        tile_info.update({"split_transform": OmegaConf.to_container(self.split_transform_dict)})
        if not self.classmapping:
            tmp_remap = {key: i + 1 for i, key in enumerate(self.data_classesid)}
            self.classmapping, self.reversal_classmapping = init_classmapping(tmp_remap)
        self.classes = list(sorted(set(self.classmapping.values())))
        tile_info["data_classesid"] = self.data_classesid
        tile_info["classmapping"] = self.classmapping
        torch.save(tile_info, self.conf_file)
        return res

    def get(self, idx):
        data = torch.load(self._scans[idx])
        if data.y is not None and self.classmapping:
            data.y = self._remap_labels(data.y)
        return data

    def len(self):
        return len(self._scans)

    def _remap_labels(self, semantic_label):
        """Remaps labels to [0 ; num_labels -1]. Can be overriden."""
        new_labels = semantic_label.clone()
        new_labels[:] = -1
        for source, target in self.classmapping.items():
            new_labels[semantic_label == int(source)] = target
        return new_labels

    def _reversal_labels(self, semantic_label):
        """Remaps labels to [0 ; num_labels -1]. Can be overriden."""
        new_labels = semantic_label.clone()
        for source, target in self.reversal_classmapping.items():
            new_labels[semantic_label == int(source)] = target
        return new_labels

    @property
    def num_classes(self):
        if -1 in self.classes:
            return len(self.classes) - 1
        else:
            return len(self.classes)

    @property
    def num_features(self) -> int:
        if self._num_features:
            return self._num_features
        else:
            return self.num_node_features


class LidarClassifyTest(Dataset):
    def __init__(
        self,
        data_path: str,
        reversal_classmapping: Dict,
        num_features: int,
        transform=None,
        pre_transform=None,
        split_transform=None,
    ):
        self.reversal_classmapping = reversal_classmapping
        super().__init__(data_path, transform=None, pre_transform=None)
        self._transform = transform
        self._pre_transform = pre_transform
        self.classes = reversal_classmapping.keys()
        self._scans = self.get_scans(data_path)
        self.split_transform = split_transform
        self._num_features = num_features

    def get_scans(self, data_path, block_size=None):
        if isinstance(data_path, str):
            path = Path(data_path)

            # Kiểm tra nếu data_path là file cụ thể
            if path.is_file() and path.suffix.lower() in ['.lidata', '.las', '.laz']:
                # Nếu là file cụ thể, chỉ trả về file đó
                scans = [path]
            else:
                # Nếu là thư mục, tìm tất cả các file hợp lệ
                lidata_files = list(Path(data_path).glob("**/*.Lidata"))
                las_files = list(Path(data_path).glob("**/*.las"))
                laz_files = list(Path(data_path).glob("**/*.laz"))
                scans = lidata_files + las_files + laz_files
                
                # Kiểm tra xem có file nào tồn tại không
                if not scans:
                    print(f"WARNING: Không tìm thấy file dữ liệu nào trong đường dẫn: {data_path}")
                    return []
        else:
            scans = [Path(_) for _ in data_path]
        
        scans_bounds = []
        for scan_file in scans:
            # Kiểm tra xem file có tồn tại không
            if not scan_file.exists():
                print(f"WARNING: File không tồn tại: {scan_file}")
                continue
            
            # Kiểm tra phần mở rộng của file
            ext = scan_file.suffix.lower()
            if ext in ['.lidata', '.las', '.laz']:
                try:
                    sub_bounds = DataReader(scan_file).sub_bounds(block_size=block_size)
                    scan_bounds = zip([scan_file for i in range(len(sub_bounds))], sub_bounds)
                    scans_bounds += scan_bounds
                except Exception as e:
                    print(f"ERROR đọc file {scan_file}: {str(e)}")
        
        return scans_bounds

    def write_res(self, raw_data, output):
        if raw_data.batch.dim() == 2:
            raw_data.batch = raw_data.batch.squeeze()
        batch = raw_data.batch.max() + 1
        index = 0
        for i in range(batch):
            file_name = raw_data.tile[i][0]
            sub_bounds = raw_data.tile[i][1]
            idx = raw_data.batch == i
            output_batch = self._reversal_labels(output[idx])
            data_header = DataReader(file_name)
            _, cls = data_header.read_boxf_lidata(sub_bounds)
            if len(cls) == len(output_batch):
                data_header.writer_boxf_lidata(sub_bounds, output.numpy().astype("u1"))
            else:
                recon = getattr(raw_data, "origin_id", None)
                if recon != None:
                    data_header.writer_boxf_lidata(
                        sub_bounds, output_batch[recon[index : index + len(cls)]].numpy().astype("u1")
                    )
                    index = index + len(cls)
                else:
                    basic_log.warning(f"wirte {file_name} failed! size dismatch!")
                    return False
            log.info("Processed file %s, points number = %i", os.path.basename(file_name), data_header.point_count)
        return True

    def write_res_split(self, raw_data, output):
        sub_bounds = raw_data.tile[1]
        output = self._reversal_labels(output)
        data_header = DataReader(raw_data.tile[0])
        
        # Kiểm tra phần mở rộng của file
        ext = raw_data.tile[0].suffix.lower()
        
        recon = getattr(raw_data, "origin_id", None)
        if recon is not None:
            if ext == '.lidata':
                # Ghi kết quả cho file LiData
                data_header.writer_boxf_lidata(sub_bounds, output[recon].numpy().astype("u1"))
            else:
                # Ghi kết quả cho file LAS/LAZ
                self._write_las_result(raw_data.tile[0], output[recon].numpy().astype("u1"))
        else:
            if ext == '.lidata':
                # Ghi kết quả cho file LiData
                data_header.writer_boxf_lidata(sub_bounds, output.numpy().astype("u1"))
            else:
                # Ghi kết quả cho file LAS/LAZ
                self._write_las_result(raw_data.tile[0], output.numpy().astype("u1"))
            return False
        
        log.info(
            f"Processed file {raw_data.tile[0].name}, points number ={data_header.sub_point_cloud}/{data_header.point_count}"
        )
        return True

    def _write_las_result(self, las_file, classification):
        """Ghi kết quả phân loại trực tiếp vào file LAS gốc"""
        las = laspy.read(las_file)
        
        las.classification = classification
        
        las.write(las_file)
        
        return True
    
    # def _write_las_result(self, las_file, classification):
    #     """Ghi kết quả phân loại vào file LAS"""
    #     las = laspy.read(las_file)
 
    #     las.classification = classification
        
    #     result_file = las_file.parent / f"result_{las_file.name}"
        
    #     las.write(result_file)
        
    #     return True

    def feature_process(self,data):
        f = data.pos.clone()
        f = self._norm(f)[:,2]
        return f
    def _norm(self,data):
        centroid = torch.mean(data, axis=0) 
        data = data - centroid 
        m = torch.max(torch.sqrt(torch.sum(data ** 2, axis=1)))
        data_normalized = data / m 
        data = (data - centroid) / m
        return data   

    def get(self, idx):
        file_path, block_size = self._scans[idx][0], self._scans[idx][1]
        
        # Kiểm tra phần mở rộng của file
        ext = file_path.suffix.lower()
        
        # Đọc dữ liệu từ file
        xyz, cls, rgb, intensity = DataReader(file_path).read_boxf_lidata(block_size)
        
        # Kiểm tra dữ liệu rỗng
        if not np.any(xyz):
            return None
        
        # Chuẩn hóa dữ liệu
        xyz = xyz - xyz.min(0)
        rgb = torch.tensor(rgb / 255, dtype=torch.float32)
        intensity = torch.tensor(intensity / 65535, dtype=torch.int32)
        
        # Tạo đối tượng Data
        data = Data(
            pos=torch.tensor(xyz, dtype=torch.float32),
            y=torch.tensor(cls, dtype=torch.long),
            rgb=rgb,
            intensity=intensity,
        )
        
        # Áp dụng pre_transform
        if self._pre_transform:
            data = self._pre_transform(data)
        
        # Áp dụng split_transform
        if self.split_transform:
            data_dict = self.split_transform(data)
            data = data_dict["data"]
            if self._transform:
                for i in range(len(data)):
                    data[i] = self._transform(data[i])
                    delattr(data[i], "origin_id")
            data = Data(split=data)
            if "origin_id" in data_dict.keys():
                data.origin_id = data_dict["origin_id"]
        else:
            if self._transform:
                data = self._transform(data)
        
        # Thêm thông tin tile
        data.tile = self._scans[idx]
        return data

    def len(self):
        return len(self._scans)

    def _reversal_labels(self, semantic_label):
        """Remaps labels to [0 ; num_labels -1]. Can be overriden."""
        new_labels = semantic_label.clone()
        for source, target in self.reversal_classmapping.items():
            new_labels[semantic_label == int(source)] = int(target)
        return new_labels

    @property
    def num_classes(self):
        if -1 in self.classes:
            return len(self.classes) - 1
        else:
            return len(self.classes)

    @property
    def num_features(self) -> int:
        return self._num_features


class LidarClassifyDataset(BaseDataset):
    """Wrapper around Lidar that creates train and test datasets.
    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain
            - root,
            - classmapping,
            - transform,
            - pre_transform
            - process_workers
    """

    def __init__(self, dataset_opt):
        if not getattr(dataset_opt, "dataroot", ""):
            setattr(dataset_opt, "dataroot", "")
        pre_transform = getattr(dataset_opt, "pre_transform", [])
        split_transform = getattr(dataset_opt, "split_transform", [])
        if isinstance(pre_transform, str):
            delattr(dataset_opt, "pre_transform")
        super().__init__(dataset_opt)
        process_workers = dataset_opt.process_workers if dataset_opt.process_workers else 0
        if process_workers > multiprocessing.cpu_count():
            basic_log.warning(f"too many workers for cpu ,will set {multiprocessing.cpu_count()}  workers")
            process_workers = multiprocessing.cpu_count()
        self.process_workers: int = process_workers
        classmapping = getattr(dataset_opt, "classmapping", None)
        classmapping = OmegaConf.to_container(classmapping)
        self.reversal_classmapping = getattr(dataset_opt, "reversal_classmapping", None)
        self.dataset_opt = dataset_opt
        if getattr(dataset_opt, "train_path", ""):
            self.train_dataset = LidarClassify(
                data_path=dataset_opt.train_path,
                classmapping=classmapping,
                transform=self.train_transform,
                pre_transform=self.pre_transform,
                split_transform=self.split_transform,
                pre_transform_dict=pre_transform,
                split_transform_dict=split_transform,
                process_workers=self.process_workers,
            )
            if not self.reversal_classmapping:
                self.reversal_classmapping = self.train_dataset.reversal_classmapping
        if getattr(dataset_opt, "val_path", ""):
            self.val_dataset = LidarClassify(
                data_path=dataset_opt.val_path,
                classmapping=classmapping,
                transform=self.val_transform,
                pre_transform=self.pre_transform,
                split_transform=self.split_transform,
                pre_transform_dict=pre_transform,
                split_transform_dict=split_transform,
                process_workers=self.process_workers,
            )
        if getattr(dataset_opt, "test_path", ""):
            reversal_classmapping = getattr(dataset_opt, "reversal_classmapping", None)
            num_features = getattr(dataset_opt, "num_features", None)
            reversal_classmapping = OmegaConf.to_container(reversal_classmapping)

            self.test_dataset = LidarClassifyTest(
                data_path=dataset_opt.test_path,
                reversal_classmapping=reversal_classmapping,
                num_features=num_features,
                transform=self.test_transform,
                pre_transform=self.pre_transform,
                split_transform=self.split_transform,
            )

    def get_tracker(self, tensorboard_log: bool):
        """Factory method for the tracker
        Arguments:
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return SegmentationTracker(self, use_tensorboard=tensorboard_log)
