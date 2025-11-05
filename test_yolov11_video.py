from ultralytics import YOLO

def main():

    yolo11n_model = "/Users/patrickryan/Development/machinelearning/yolo-sandbox/yolo11-custom-object-detection/yolo11n.pt"
    custom_yolo_model = "/Users/patrickryan/Development/machinelearning/yolo-sandbox/yolo11-custom-object-detection/yolo11n_custom/yolo11n_custom_run1/weights/best.pt"
    model = YOLO(custom_yolo_model)


    results = model.predict(source="/Users/patrickryan/Development/machinelearning/yolo-sandbox/yolo11-custom-object-detection/test_videos/dexi_camera_all_classes_85_compression.mp4",
                            show=True,
                            save=True,
                            line_width=2,
                            save_txt=False,
                            save_crop=False,
                            show_labels=True
                            )


if __name__ == '__main__':
    main()
