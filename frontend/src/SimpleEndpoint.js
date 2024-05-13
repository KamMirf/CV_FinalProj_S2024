import React, { useState } from 'react';

const SimpleEndpointButton = () => {
  const [message, setMessage] = useState('');

  const handleButtonClick = async () => {
    try {
      const response = await fetch('http://localhost:5001/simple-endpoint', {
        method: 'GET',
      });
      if (response.ok) {
        const data = await response.text();
        setMessage(data);
      } else {
        setMessage('Error: Unable to reach the endpoint');
      }
    } catch (error) {
      console.error('Error fetching data:', error);
      setMessage('Error: An unexpected error occurred');
    }
  };

  return (
    <div>
      <button onClick={handleButtonClick}>Test Simple Endpoint</button>
      {message && <p>{message}</p>}
    </div>
  );
};

export default SimpleEndpointButton;
