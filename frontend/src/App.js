import React from 'react';
import './App.css';
import PhotoUpload from './PhotoUpload';
import RecipeGen from './RecipeGen';
import Menu from './Menu';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>HUNGRY HIPPOS</h1>
        <p>If you are out of ideas on what to make for dinner and have an assortment of food in your fridge, have no fear!</p>
        <p>First choose a model and an image, either a preloaded one or upload your own.</p>
        <Menu />
        <PhotoUpload />
        <p>Now go ahead and generate a recipe!</p>
        <RecipeGen />
      </header>
    </div>
  );
}

export default App;
