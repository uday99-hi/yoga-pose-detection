# YogaIntelliJ - Setup and Run Guide

This is a React-based Yoga Pose Detection application that uses TensorFlow.js to detect and classify yoga poses in real-time using your webcam.

## Project Structure

- **frontend/**: React application (main application)
- **classification model/**: Python scripts for training the ML model (optional - model is already trained)

## Prerequisites

1. **Node.js** (version 14 or higher recommended)
   - Download from: https://nodejs.org/
   - Verify installation: `node --version` and `npm --version`

2. **Modern Web Browser** (Chrome, Firefox, Edge, or Safari)
   - Webcam access required for pose detection

## Quick Start

### Step 1: Install Dependencies

Navigate to the frontend directory and install dependencies:

```powershell
cd "F:\final project\YogaIntelliJ-main\frontend"
npm install
```

### Step 2: Start the Application

Run the development server:

```powershell
npm start
```

The application will automatically open in your browser at `http://localhost:3000`

If it doesn't open automatically, manually navigate to: **http://localhost:3000**

## Application Features

- **Home Page**: Landing page with project information
- **Yoga Practice**: Real-time pose detection and classification
- **Tutorials**: Instructions and guides
- **About**: Project information

## Available Poses

The application can detect the following yoga poses:
- Tree
- Chair
- Cobra
- Warrior
- Dog (Downward Dog)
- Shoulderstand
- Triangle

## Troubleshooting

### Issue: `react-scripts` not recognized
**Solution**: 
```powershell
cd frontend
npm install
```

### Issue: Port 3000 already in use
**Solution**: The app will prompt you to use a different port, or you can stop the process using port 3000.

### Issue: Webcam not working
**Solution**: 
- Ensure your webcam is connected and working
- Grant camera permissions in your browser
- Check browser settings for camera access

### Issue: Model files missing
**Solution**: The model files (`model.json` and `group1-shard1of1.bin`) should be in `frontend/src/`. If missing, they need to be regenerated from the classification model.

## Development Notes

- The app uses TensorFlow.js for client-side pose detection
- No backend server is required - everything runs in the browser
- The trained model is already included in the frontend/src folder
- The classification model folder contains Python code for training/retraining the model (optional)

## Scripts Available

- `npm start`: Start development server
- `npm build`: Build for production
- `npm test`: Run tests
- `npm eject`: Eject from Create React App (irreversible)

## Browser Compatibility

- Chrome (recommended)
- Firefox
- Edge
- Safari

Note: Some features may require a modern browser with WebGL support for TensorFlow.js.


