const express = require('express');
const axios = require('axios');
const path = require('path');
const { spawn } = require('child_process');

const OpenAI = require('openai');

const openai = new OpenAI({
  // apiKey: process.env.OPENAI_KEY
  apiKey: 'sk-proj-ZDvnzjHLfBw9EiZ8QgE6T3BlbkFJhjcBTALxsHHWbh0h01IH'
});


require('dotenv').config();

const multer = require('multer');
const { exec } = require('child_process');
const app = express();
app.use(express.json());

const cors = require('cors');
const bodyParser = require('body-parser');
app.use(cors());

// Set up multer to handle file uploads
const upload = multer({ dest: 'uploads/' });

const PORT = process.env.PORT || 5001; // backend server

app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
  next();
});

let fetch;

(async () => {
  fetch = (await import('node-fetch')).default;
})();

// Define a simple route to ensure server is running
app.get('/', (req, res) => {
  res.send('Hello World!');
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

let concatenatedKeys = "";

//Handle yolo model
app.post('/api/upload/yolo', upload.single('photo'), (req, res) => {
  
  const file = req.file;
  if (!file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }
  
  // Process the uploaded image (e.g., with your ML model)
  // Execute Python script with the uploaded image as an argument
  const pythonScriptDir = path.join(__dirname, 'yolo-code');
  process.chdir(pythonScriptDir);

  exec(`python run-yolov5-on-image.py ${file.path}`, (error, stdout, stderr) => {
    if (error) {
      console.error('Error executing Python script:', error);
      res.status(500).json({ error: 'Internal server error' });
      return;
    }
    console.log('Python script output:', stdout);
    // Process output from Python script if needed

    concatenatedKeys = ''

    // Loop through the keys of the object
    for (let key in stdout) {
        // Check if the key is a string
        if (typeof key === 'string') {
            // Concatenate the key to the existing string
            concatenatedKeys += key + " ";
        }
    }
    res.json({ message: 'Image processed successfully', output: stdout });
  });
});

//Handle custom model
app.post('/api/upload/custom', upload.single('photo'), (req, res) => {
  
  const file = req.file;
  if (!file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }
  
  // Process the uploaded image (e.g., with your ML model)
  // Execute Python script with the uploaded image as an argument
  const pythonScriptDir = path.join(__dirname, 'classifier_detector_code');
  process.chdir(pythonScriptDir);

  exec(`python extract_results.py ${file.path}`, (error, stdout, stderr) => {
    if (error) {
      console.error('Error executing Python script:', error);
      res.status(500).json({ error: 'Internal server error' });
      return;
    }
    console.log('Python script output:', stdout);

    concatenatedKeys = ''

    // Loop through the keys of the object
    for (let key in stdout) {
      // Check if the key is a string
      if (typeof key === 'string') {
          // Concatenate the key to the existing string
          concatenatedKeys += key + " ";
      }
    }
    // Process output from Python script if needed
    res.json({ message: 'Image processed successfully', output: stdout });
  });
});

// Additional route to handle GPT API calls
app.post('/get-recipe', async (req, res) => {
  const prompt = "Give me a recipe using some of these ingredients: "; // prompt for a random recipe
  // const prompt = "Give me a recipe using these ingredients: "
  try {

    const completion = await openai.chat.completions.create({
      messages: [
        {"role": "system",
        "content": prompt + concatenatedKeys}
      ],
      model: "gpt-3.5-turbo",
    });

    data = completion.choices[0].message.content
    prompt_and_comp = prompt + data

    console.log(prompt_and_comp)

    res.status(200).send(prompt_and_comp); 

    
  } catch (error) {
      console.error('Failed to fetch from OpenAI:', error);
      res.status(500).json({ error: error.message });
    }

  });

  app.get('/simple-endpoint', (req, res) => {
    res.status(200).send("endpoint working");
  });