const express = require('express');
const axios = require('axios');

// const openai = new OpenAIApi(config)

require('dotenv').config();

const app = express();
app.use(express.json());

const cors = require('cors');
const bodyParser = require('body-parser');
app.use(cors());

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


// Additional route to handle GPT API calls
app.post('/get-recipe', async (req, res) => {
  const prompt = "Give me a random recipe"; // Static prompt for a random recipe
  const apiKey = process.env.OPENAI_KEY;
  // const client = axios.create({
  //   headers: { 'Authorization': 'Bearer ' + apiKey }
  // });
  // const params = {
  //   "prompt": prompt, 
  //   "max_tokens": 10
  // }
  
  try {
    // const response = "hello"
    const response = await fetch('https://api.openai.com/v1/engines/gpt-3.5-turbo/completions', {
    // const response = await openai.createCompletion({
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        prompt: prompt,
        max_tokens: 150
      })
    });


    if (!response.ok) {
      throw new Error('Network response was not OK');
    }

    const data = await response.json();
    console.log(data)

    // res.status(200).json(data.choices[0].text);
    res.status(200).send(data.choices[0].text);


    
  } catch (error) {
      console.error('Failed to fetch from OpenAI:', error);
      res.status(500).json({ error: error.message });
    }

  });

  app.get('/simple-endpoint', (req, res) => {
    res.status(200).send("endpoint working");
  });