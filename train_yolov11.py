from ultralytics import YOLO

"""
Training script for YOLOv11 model on custom dataset.

This training can be done on the command line.  

See:  https://docs.ultralytics.com/modes/train/

For example

yolo detect train data=./modeling_data/dataset_custom.yaml model=yolo11n.pt epochs=10 imgsz=640 device='mps'


"""
def main():
    model = YOLO('yolo11n.pt')
    results = model.train(data='./modeling_data/dataset_custom.yaml', epochs=10, imgsz=640, batch=16, device='mps',
                          project='yolo11n_custom', name='yolo11n_custom_run1')
    return results

if __name__ == '__main__':
    model_train_results = main()
    print("Done!")
    print("*"*20)
    print(f"Model saved to: {model_train_results.model.path}")