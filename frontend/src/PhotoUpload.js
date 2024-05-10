import React, { useState } from 'react';

function PhotoUpload() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadResponse, setUploadResponse] = useState(null);

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

    // Make sure to replace 'YOUR_UPLOAD_URL' with your actual upload endpoint URL
    fetch('http://localhost:5001/api/upload', {
      method: 'POST',
      body: formData,
    })
    .then(response => {
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      return response.json();
    })
    .then(data => {
      console.log('Success:', data);
      setUploadResponse(data);
      alert('Upload successful!');
    })
    .catch((error) => {
      console.error('Error:', error);
      alert('Upload failed!');
    });
  };

  return (
    <div className="form-section">
      <input type="file" onChange={handleFileChange} accept="image/*" className="form-control" />
      {selectedFile && <p className="file-info">File selected: {selectedFile.name}</p>}
      {uploadResponse && ( // Render response data if it exists
        <div>
          <h3>Upload Response:</h3>
          <pre>{JSON.stringify(uploadResponse, null, 2)}</pre> {/* Render response data */}
        </div>
      )}
    </div>
  );
}

export default PhotoUpload;
