import logo from './logo.svg';
import './App.css';
import PhotoUpload from './PhotoUpload'; 

function App() {
  return (
    <div className="App">
      <header className="App-header">

        <h2>
          Welcome to our CV Final Project: Hungry Hippos!
        </h2>

        <p>If you are out of ideas on what to make for dinner and have assortment of food in your fridge, have no fear!</p>

        <p>Upload a photo of your fridge, and we will provide you with a recipe that you can make with the items in your fridge.</p>

        <PhotoUpload />
        
      </header>
    </div>


  );
}

export default App;
