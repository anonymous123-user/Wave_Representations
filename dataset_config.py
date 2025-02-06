DATASET_CONFIG = {
    'new_tetronimoes': {
        'x_train_path': "data/new_tetrominoes/train_images.npy",
        'y_train_path': "data/new_tetrominoes/train_masks.npy",
        'x_val_path': "data/new_tetrominoes/val_images.npy",
        'y_val_path': "data/new_tetrominoes/val_masks.npy",
        'x_test_path': "data/new_tetrominoes/test_images.npy",
        'y_test_path': "data/new_tetrominoes/test_masks.npy",
        'img_size': 64,
        'channels': 3,
    },
    'mnist': {
        'train_path': 'data/mnist/',
        'test_path': 'data/mnist/',
        'img_size': 56,  # Image size for resizing
        'channels': 1,
    },
}