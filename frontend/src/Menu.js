import React, { useState } from 'react';

const Menu = () => {
    // State for radio buttons
    const [selectedModel, setSelectedModel] = useState('model1');

    // State for dropdown
    const [selectedImage, setSelectedImage] = useState('');

    // Handle radio button change
    const handleModelChange = (event) => {
        setSelectedModel(event.target.value);
    };

    // Handle dropdown change
    const handleImageChange = (event) => {
        setSelectedImage(event.target.value);
    };

    // Example image options (replace with actual image data)
    const imageOptions = [...Array(20).keys()].map((i) => ({
        id: `image${i + 1}`,
        name: `Image ${i + 1}`,
    }));

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
        </div>
    );
};

export default Menu;
