# Streamlit Set Game Detector

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-brightgreen)](your-streamlit-app-link-here)

## Project Overview

The Streamlit Set Game Detector is a web application designed to automatically identify sets in images of Set card game boards. Built using Python, this application utilizes AI models to streamline the process of set detection.  Users can upload an image of a Set game layout, and the application will process it to highlight all valid sets present. This tool is intended to assist players in learning the game, verifying set identification, or simply for automated set discovery.

## Key Features

*   **Automated Set Identification:**  Accurately detects and highlights all valid sets within a provided image of a Set game board.
*   **Image Orientation Handling:**  Automatically adjusts for image orientation, ensuring accurate processing regardless of how the image is captured.
*   **Visual Set Highlighting:**  Clearly annotates the input image by drawing bounding boxes and labels around detected sets for easy visual confirmation.
*   **Streamlit Web Interface:**  Provides a user-friendly web interface built with Streamlit, accessible via any standard web browser.
*   **AI-Powered Detection:**  Employs YOLOv8 for card and shape detection and Keras models for classifying essential card attributes (color, fill, shape).

## Technologies

The application is built upon the following technologies:

*   **Python:**  Primary programming language for application logic and AI model implementation.
*   **Streamlit:**  Framework for creating the interactive web application.
*   **OpenCV (cv2):**  Library used for image processing tasks, including image manipulation and annotation.
*   **NumPy:**  Fundamental library for numerical computations, particularly for image data handling.
*   **Pandas:**  Library for data manipulation and analysis, used for organizing card feature data.
*   **TensorFlow/Keras:**  Deep learning framework used to build and deploy models for card attribute classification (shape, fill).
*   **Ultralytics YOLOv8:**  Object detection framework utilized for identifying cards and shapes within images.
*   **PyTorch:**  Backend framework for YOLOv8, enabling GPU acceleration for model inference.

## Getting Started

To run the application locally, follow these installation and setup steps:

1.  Clone the Repository
2.  Set Up Environment and Install Dependencies
3.  Download Pre-trained Models:
    Download the pre-trained models for shape classification, fill classification, and YOLOv8 object detection. Place these model folders in a directory named `models` at the repository root, or adjust the `base_dir` path in `app.py` to reflect your model directory location. The expected directory structure should be:

    ```
    set-game-detector/
    ├── app.py
    ├── models/
    │   ├── Card/
    │   │   └── 16042024/
    │   │       ├── best.pt
    │   │       └── data.yaml
    │   ├── Characteristics/
    │   │   └── 11022025/
    │   │       ├── fill_model.keras
    │   │       └── shape_model.keras
    │   └── Shape/
    │       └── 15052024/
    │           └── best.pt
    │           └── data.yaml
    ├── README.md
    ├── requirements.txt
    └── ... (other files)
    ```

## Contributing

Contributions to this project are welcome. To contribute:

1.  Fork the repository.
2.  Create a branch for your feature or bug fix.
3.  Implement your changes and commit them.
4.  Submit a pull request detailing the changes you've made.

## License

[MIT License](LICENSE)

## Contact Information

omamitai - omermamitai@gmail.com

---

I hope this tool is helpful for Set game enthusiasts and developers alike.
