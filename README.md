# SculptVerse

## Setup

### Installation
```
git clone https://github.com//hardik4tiwari/SculptVerse.git
cd SculptVerse
```

### Environment
- Install requirements for SculptVerse first.
  ```
  pip install -r requirements.txt
  ```

## Quick Start

### Prepare Images
- We put some sample inputs under `assets/sample_input`, and you can quickly try them.
- Prepare RGBA images or RGB images with white background (with some background removal tools, e.g., [Rembg](https://github.com/danielgatis/rembg), [Clipdrop](https://clipdrop.co)).
- An example usage of preprocessing is as follows:
  ```
  #Example Usage
  IMAGE_INPUT="./assets/sample_input/box.jpg"
  IMG_PTH="INPUT/box.jpg"
  python preprocess_images.py $IMG_PTH $IMAGE_INPUT --rmbg --recenter
  ```

### Inference
- Run the inference script to get 3D assets.
- You may specify which form of output to generate by setting the flags `EXPORT_VIDEO=true` and `EXPORT_MESH=true`.
- An example usage is as follows:

  ```
  # Example usage
  EXPORT_VIDEO=true
  EXPORT_MESH=true
  INFER_CONFIG="./configs/infer-b.yaml"
  MODEL_NAME="zxhezexin/openlrm-mix-base-1.1"
  IMAGE_INPUT="./assets/sample_input/owl.png"

  python -m openlrm.launch infer.lrm --infer $INFER_CONFIG model_name=$MODEL_NAME image_input=$IMAGE_INPUT export_video=$EXPORT_VIDEO export_mesh=$EXPORT_MESH
  ```

  ### OBJ File
  - Run the conversion script to get obj model
  - An example usage is as follows:

    ```
    INPUT_FILE="path/to/input_file.ply"
    OUTPUT_FILE="path/to/output_file.obj"
    python convert_ply_to_obj.py $INPUT_FILE $OUTPUT_FILE
    ```
