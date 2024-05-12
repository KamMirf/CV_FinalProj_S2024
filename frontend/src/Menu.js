import React, { useState } from 'react';

const Menu = () => {
    // State for radio buttons
    const [selectedModel, setSelectedModel] = useState('model1');

    // State for dropdown
    const [selectedImage, setSelectedImage] = useState('');

	// State for uploaded image
	const [selectedFile, setSelectedFile] = useState(null);

    // State for response from upload
    const [uploadResponse, setUploadResponse] = useState(null);

    // Handle radio button change
    const handleModelChange = (event) => {
        setSelectedModel(event.target.value);
    };

    const [displayedImagePath, setDisplayedImagePath] = useState('');

    // Handle dropdown change
    const handleImageChange = (event) => {
        const path = event.target.value;

        const modifiedPath = path.replace('..','../backend')
        console.log(modifiedPath)
        setDisplayedImagePath(modifiedPath);
        setSelectedImage(event.target.value);
        let uploadURL
        if (selectedModel === 'model1') {
            uploadURL = 'http://localhost:5001/api/upload/yolo'
        }
        else {
            uploadURL = 'http://localhost:5001/api/upload/custom'
        }
        // Send selected image path to backend
        fetch(uploadURL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ imagePath: selectedImage }),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            console.log('Success:', data);
            // Handle response from backend if needed
        })
        .catch((error) => {
            console.error('Error:', error);
            // Handle error
        });

    };

	// Handle upload image change
	const handleFileChange = (event) => {
		const file = event.target.files[0];
		if (file) {
		  setSelectedFile(file);
		  uploadFile(file); // Automatically upload the file once selected
		}
	  };

    const imagePaths = [
        '../classifier_detector_code/data/valid/images/DSC_5677_JPG_jpg.rf.bdbf6734f9a2eaf1b2438e844cc439c9.jpg',
        '../classifier_detector_code/data/valid/images/DSC_5678_JPG_jpg.rf.88e59a6421653b332a514c34d942237a.jpg',
        '../classifier_detector_code/data/valid/images/DSC_5679_JPG_jpg.rf.82566b78b96aa89388dee0caff7dbcae.jpg',
        '../classifier_detector_code/data/valid/images/DSC_5683_JPG_jpg.rf.59ee5ce9c2ba75c699c67254fd20bc72.jpg',
        '../classifier_detector_code/data/valid/images/DSC_5686_JPG_jpg.rf.54ec15556ab9c5119abe4d32ffa34709.jpg',
        '../classifier_detector_code/data/valid/images/DSC_5703_JPG_jpg.rf.abf86d888d179142f7dc44975006fdd6.jpg',
        '../classifier_detector_code/data/valid/images/DSC_5704_JPG_jpg.rf.f21034ccb0030313527db49b8afcbc4f.jpg',
        '../classifier_detector_code/data/valid/images/DSC_5705_JPG_jpg.rf.e7db4657ab081d4afc1cbc4bb3803450.jpg',
        '../classifier_detector_code/data/valid/images/DSC_5730_JPG_jpg.rf.b9bff07bf9746022b7e2a45f750d3e6f.jpg',
        '../classifier_detector_code/data/valid/images/DSC_5733_JPG_jpg.rf.b5f5e858cb9bd1bca1ab436e096987be.jpg'
    ];
    
    // Create dropdown options using image paths
    const imageOptions = imagePaths.map((path, index) => ({
    id: `image${index + 1}`,
    name: `Image ${index + 1}`,
    path: path, 
    }));

	// upload image functionality - if we choose to implement it at a later data
	const uploadFile = (file) => {
        let uploadURL
        if (selectedModel === 'model1') {
            uploadURL = 'http://localhost:5001/api/upload/yolo'
        }
        else {
            uploadURL = 'http://localhost:5001/api/upload/custom'
        }
        const formData = new FormData();
        formData.append('photo', file);
    
        fetch(uploadURL, {
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
                            <option key={option.id} value={option.path}>
                                {option.name}
                            </option>
                        ))}
                    </select>

                    {/* Display the selected image */}
                    {selectedImage && (
                        <div>
                            <img src={displayedImagePath} />
                        </div>
                    )}

                </label>
            </div>


        </div>
    );
};

export default Menu;
