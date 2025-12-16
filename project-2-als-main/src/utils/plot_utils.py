import matplotlib.pyplot as plt

def show_in_row(images, titles=None, figsize=(15, 5), wspace=0.025, bg_color='#D3D3D3'):
    """
    Displays a list of images in a single row.
    
    Args:
        images: List of numpy arrays (images).
        titles: List of strings (optional).
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize, facecolor=bg_color)
    if n == 1: axes = [axes]
    
    for i in range(n):
        cmap = 'gray' if len(images[i].shape) == 2 else None
        
        axes[i].imshow(images[i], cmap=cmap)
        axes[i].axis('off') 
        
        if titles and i < len(titles):
            axes[i].set_title(titles[i])
            
    plt.subplots_adjust(wspace=wspace)
    plt.show()
