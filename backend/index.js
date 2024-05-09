const express = require('express');
const axios = require('axios');

const OpenAI = require('openai');

const openai = new OpenAI({
  apiKey: process.env.OPENAI_KEY
});


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


//  route to handle GPT API calls
app.post('/get-recipe', async (req, res) => {
  const prompt = "Give me a random recipe"; // prompt for a random recipe
  // const prompt = "Give me a recipe using these ingredients: "
  try {

    const completion = await openai.chat.completions.create({
      messages: [
        {"role": "system",
        "content": prompt}
      ],
      model: "gpt-3.5-turbo",
    });

    data = completion.choices[0].message.content

    // console.log(data)

    res.status(200).send(data); 

    
  } catch (error) {
      console.error('Failed to fetch from OpenAI:', error);
      res.status(500).json({ error: error.message });
    }

  });

  app.get('/simple-endpoint', (req, res) => {
    res.status(200).send("endpoint working");
  });