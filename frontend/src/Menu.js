import React, { useState } from 'react';

const Menu = () => {
    // State for radio buttons
    const [selectedModel, setSelectedModel] = useState('model1');

    // State for dropdown
    const [selectedImage, setSelectedImage] = useState('');

	// State for uploaded image
	const [selectedFile, setSelectedFile] = useState(null);

    // Handle radio button change
    const handleModelChange = (event) => {
        setSelectedModel(event.target.value);
    };

    // Handle dropdown change
    const handleImageChange = (event) => {
        setSelectedImage(event.target.value);
    };

	// Handle upload image change
	const handleFileChange = (event) => {
		const file = event.target.files[0];
		if (file) {
		  setSelectedFile(file);
		  uploadFile(file); // Automatically upload the file once selected
		}
	  };

    // Example image options (replace with actual image data)
    const imageOptions = [...Array(20).keys()].map((i) => ({
        id: `image${i + 1}`,
        name: `Image ${i + 1}`,
    }));

	// upload image functionality
	const uploadFile = (file) => {
		const formData = new FormData();
		formData.append('photo', file);
	
		// Make sure to replace 'YOUR_UPLOAD_URL' with your actual upload endpoint URL
		fetch('YOUR_UPLOAD_URL', {
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
		  alert('Upload successful!');
		})
		.catch((error) => {
		  console.error('Error:', error);
		  alert('Upload failed!');
		});
	  };

    return (
        <div className="form-section">
            {/* Radio Button Section */}
            <div className="radio-container">
                <label>
                    <input
                        type="radio"
                        className="radio-input"
                        value="model1"
                        checked={selectedModel === 'model1'}
                        onChange={handleModelChange}
                    />
                    YOLO
                </label>
                <label style={{ marginLeft: '10px' }}>
                    <input
                        type="radio"
                        className="radio-input"
                        value="model2"
                        checked={selectedModel === 'model2'}
                        onChange={handleModelChange}
                    />
                    Custom Model
                </label>
            </div>

            {/* Dropdown Section */}
            <div className="dropdown mt-20">
                <label>
                    <select value={selectedImage} onChange={handleImageChange} className="form-control">
                        <option value="">--Select an image--</option>
                        {imageOptions.map((option) => (
                            <option key={option.id} value={option.id}>
                                {option.name}
                            </option>
                        ))}
                    </select>
                </label>
            </div>

			{/* Upload Section*/ }
			<div>
			<input type="file" onChange={handleFileChange} accept="image/*" className="form-control" />
     		{selectedFile && <p className="file-info">File selected: {selectedFile.name}</p>}
			</div>

        </div>
    );
};

export default Menu;
