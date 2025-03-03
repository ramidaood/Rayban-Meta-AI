# AI-Powered Navigation Assistance for the Visually Impaired

## **Project Overview**

This project leverages Ray-Ban Meta smart glasses equipped with a live camera feed and spatial audio to assist blind and visually impaired individuals. The system integrates AI-driven obstacle detection and navigation guidance through localized audio cues to enhance user mobility and independence.

## **Key Features**

- **Real-Time Obstacle Detection**: Uses AI to recognize and identify objects in the environment.
- **Text Reading**: Reads text from signs, menus, and documents using OCR.
- **Color and Object Recognition**: Identifies objects and colors for better decision-making.
- **Spatial Audio Guidance**: Provides intuitive left/right navigation cues based on detected obstacles.
- **Live Feed Processing**: Captures real-time video from the glasses and processes it on a backend server.
- **Cloud-Enabled Backend**: A fast, scalable system for AI inference and data processing.

## **Technology Stack**

### **Hardware**

- Ray-Ban Meta Smart Glasses
- Mobile Device for Communication

### **Software & Libraries**

- **Backend**: Python (Flask, FastAPI), REST APIs, WebSockets
- **AI Models**: OpenCV, TensorFlow, YOLO (for object detection)
- **OCR**: Tesseract, Google Vision API
- **Cloud Services**: AWS/GCP/Azure for hosting
- **Communication**: WhatsApp/Messenger API

## **Project Timeline**

This project is currently in the **Backend Research Phase**.
The full development plan is structured between **February 10, 2025 – August 10, 2025**.

### **Milestones:**

1. **Backend Research & API Integration** (March 2025)
2. **OCR & Environment Recognition Development** (April 2025)
3. **AI Model Training for Object and Color Recognition** (May 2025)
4. **Backend & AI Integration** (June 2025)
5. **Testing and Optimization** (July 2025)
6. **Final Deployment & Documentation** (August 2025)

## **How to Run**

### **1. Setup Backend**

- Install dependencies: `pip install flask fastapi opencv-python pytesseract tensorflow`
- Run the backend server: `python server.py`

### **2. Connect to Glasses**

- Enable live streaming on Ray-Ban Meta glasses.
- Authenticate with the WhatsApp/Messenger API.

### **3. User Interaction**

- Start navigation mode.
- The system provides real-time feedback using spatial audio.

## **Contributors**

- **Mariana Dakwar** ([marianadakwar@gmail.com](mailto\:marianadakwar@gmail.com))
- **Rami Daood** ([daoodrami52@gmail.com](mailto\:daoodrami52@gmail.com))

## **License**

This project is intended for academic purposes and is open for accessibility-related research.

For additional questions, please contact the contributors or the industry supervisor **Shiri Hochman** at **[shiri@migdalor.org.il](mailto\:shiri@migdalor.org.il)**.

