import cv2
import numpy as np

def post_process_mask(mask, min_size=50, close_ksize=5):
    """
    Refines the binary mask by connecting gaps and removing small noise.
    
    Args:
        mask (numpy.ndarray): Binary mask (0 or 1).
        min_size (int): Minimum size (in pixels) for a blob to be kept.
        close_ksize (int): Kernel size for morphological closing (connects gaps).
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_closed, connectivity=8)
    
    final_mask = np.zeros_like(mask_closed)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area >= min_size:
            final_mask[labels == i] = 255
            
    return (final_mask > 0).astype(np.float32)