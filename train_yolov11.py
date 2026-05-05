from ultralytics import YOLO
from pathlib import Path

"""
Training script for YOLOv11 model on custom dataset.

This training can be done on the command line.  

See:  https://docs.ultralytics.com/modes/train/

For example

yolo detect train data=./modeling_data/dataset_custom.yaml model=yolo11n.pt epochs=10 imgsz=640 device='mps'


"""
yolo_model_number = 26
def main():
    BASE_DIR = Path(__file__).resolve().parent
    print(f"BASE_DIR: {BASE_DIR}")

    project = f"{BASE_DIR}/yolo{yolo_model_number}n_custom"
    run_name = f"yolo{yolo_model_number}n_custom_run1"

    model = YOLO(f'yolo{yolo_model_number}n.pt')
    results = model.train(data=f'{BASE_DIR}/modeling_data/dataset_custom.yaml', epochs=35, imgsz=640, batch=16, device='mps',
                          project=project, name=run_name,
                          save=True,
                          exist_ok=True)

    save_dir = Path(model.trainer.save_dir)
    weights_dir = save_dir / "weights"
    best_model_path = weights_dir / "best.pt"
    last_model_path = weights_dir / "last.pt"

    return results, best_model_path, last_model_path, save_dir

if __name__ == '__main__':
    model_train_results, best_model_path, last_model_path, save_dir = main()
    print("Done!")
    print("*"*20)
    print(f"Actual Ultralytics save directory: {save_dir}")
    print(f"Best model expected at: {best_model_path}")
    print(f"Last model expected at: {last_model_path}")
    print(f"best.pt exists: {best_model_path.exists()}")
    print(f"last.pt exists: {last_model_path.exists()}")

    if hasattr(model_train_results, "results_dict"):
        print("Training metrics:")
        for key, value in model_train_results.results_dict.items():
            print(f"{key}: {value}")