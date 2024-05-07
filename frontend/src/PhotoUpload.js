import React, { useState } from 'react';

function PhotoUpload() {
  const [selectedFile, setSelectedFile] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
	if (file) {
		setSelectedFile(file);
		uploadFile(file); // Automatically upload the file once selected
	  }
  
  };

  const uploadFile = (file) => {
    const formData = new FormData();
    formData.append('photo', file);

    fetch('YOUR_UPLOAD_URL', { // send to backend to pass through model
      method: 'POST',
      body: formData,
    })
    .then(response => response.json())
    .then(data => {
      console.log('Success:', data);
      alert('Upload successful!');
    })
    .catch((error) => {
      console.error('Error:', error);
      alert('Upload failed!');
    });
  };

  return (
    <div>
      <input type="file" onChange={handleFileChange} accept="image/*" />
      {selectedFile && <p>File selected: {selectedFile.name}</p>}
    </div>
  );
}

export default PhotoUpload;