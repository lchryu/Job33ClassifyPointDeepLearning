"""
Module đơn giản để test accuracy bằng cách so sánh:
- Folder chứa .LiData gốc (có ground truth labels)  
- Folder chứa .LiData đã được model phân lớp (predictions)
"""

import numpy as np
import gvlib
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def read_lidata_labels(lidata_path):
    """Đọc file LiData và chỉ trả về classification labels"""
    try:
        reader = gvlib.LidataReader(str(lidata_path))
        if not reader.open():
            raise IOError(f"Không thể mở file {lidata_path}")
        
        reader.read()
        lidata = reader.tile()
        
        # Chỉ lấy classification
        classification = lidata.classification
        return classification
        
    except Exception as e:
        print(f"Lỗi khi đọc file {lidata_path}: {str(e)}")
        raise


def calculate_accuracy(ground_truth_folder: str, predictions_folder: str) -> dict:
    """
    Tính accuracy bằng cách stack tất cả labels từ nhiều file
    
    Args:
        ground_truth_folder: Folder chứa file .LiData gốc (có labels thật)
        predictions_folder: Folder chứa file .LiData đã được model predict
    
    Returns:
        dict: Kết quả accuracy và thống kê
    """
    
    ground_truth_path = Path(ground_truth_folder)
    predictions_path = Path(predictions_folder)
    
    # Tìm tất cả file .LiData
    gt_files = sorted(list(ground_truth_path.glob("*.LiData")))
    
    print(f"Tìm thấy {len(gt_files)} file .LiData")
    
    if len(gt_files) == 0:
        raise ValueError("Không tìm thấy file .LiData trong folder ground truth")
    
    # Lists để stack tất cả labels
    all_y_true = []
    all_y_pred = []
    file_stats = []
    
    print("Đang xử lý các file...")
    print("-" * 50)
    
    # Lặp qua từng file và stack labels
    for i, gt_file in enumerate(gt_files):
        # Tìm file prediction tương ứng
        pred_file = predictions_path / gt_file.name
        
        if not pred_file.exists():
            print(f"❌ Không tìm thấy file prediction: {gt_file.name}")
            continue
        
        print(f"📁 [{i+1}/{len(gt_files)}] {gt_file.name}")
        
        try:
            # Đọc labels
            y_true = read_lidata_labels(gt_file)
            y_pred = read_lidata_labels(pred_file)
            
            # Kiểm tra số điểm
            if len(y_true) != len(y_pred):
                print(f"   ⚠️  Số điểm không khớp: GT={len(y_true):,}, Pred={len(y_pred):,}")
                # Lấy số điểm nhỏ hơn
                min_len = min(len(y_true), len(y_pred))
                y_true = y_true[:min_len]
                y_pred = y_pred[:min_len]
            
            # Stack vào lists tổng
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            
            # Lưu thống kê file
            file_stats.append({
                'filename': gt_file.name,
                'num_points': len(y_true),
                'unique_gt_classes': len(np.unique(y_true)),
                'unique_pred_classes': len(np.unique(y_pred))
            })
            
            print(f"   ✅ {len(y_true):,} points")
            
        except Exception as e:
            print(f"   ❌ Lỗi: {e}")
            continue
    
    if len(all_y_true) == 0: raise ValueError("Không có dữ liệu nào được xử lý thành công")
    
    print("-" * 50)
    print("🔄 Đang tính metrics...")
    
    # Convert to numpy arrays
    y_true_array = np.array(all_y_true)
    y_pred_array = np.array(all_y_pred)
    
    # Tính các metrics
    overall_accuracy = accuracy_score(y_true_array, y_pred_array)
    # Tính precision, recall, f1 với average='weighted' để xử lý multi-class
    precision = precision_score(y_true_array, y_pred_array, average='weighted', zero_division=0)
    recall = recall_score(y_true_array, y_pred_array, average='weighted', zero_division=0)
    f1 = f1_score(y_true_array, y_pred_array, average='weighted', zero_division=0)
    
    # Chỉ lưu thông tin cần thiết
    dicts_results = {
        'overall_accuracy': overall_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'total_points': len(y_true_array),
        'files_processed': len(file_stats),
    }
    
    return dicts_results

def print_results(results):
    print("\n=== Kết quả đánh giá ===")
    print(f"🎯 Accuracy: {results['overall_accuracy']:.4f} ({results['overall_accuracy']*100:.2f}%)")
    print(f"📏 Precision: {results['precision']:.4f} ({results['precision']*100:.2f}%)")
    print(f"📐 Recall: {results['recall']:.4f} ({results['recall']*100:.2f}%)")
    print(f"📊 F1-score: {results['f1_score']:.4f} ({results['f1_score']*100:.2f}%)")
    print(f"📈 Total points: {results['total_points']:,}")


# Main execution
if __name__ == "__main__":
    ground_truth_folder = "D:\DATA_QA\QA_test"                      # Folder chứa file gốc có labels
    predictions_folder = "D:\DATA_QA\QA_test_classify"              # Folder chứa file đã predict
    
    # Tính accuracy
    print_results(calculate_accuracy(ground_truth_folder, predictions_folder))