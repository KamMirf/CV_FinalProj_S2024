import React, { useState } from 'react';

function RecipeGen() {
  const [recipe, setRecipe] = useState('');

  const fetchRecipe = async () => {
    try {
      const response = await fetch('http://localhost:5001/get-recipe', { // matching the backend route
        method: 'POST', 
        headers: {
          'Content-Type': 'application/json'
        }
      });
      const data = await response.json(); // Assuming data comes as { recipeText: "Recipe info" }
      console.log(data); // Add this line to check what 'data' looks like

      setRecipe(data.recipeText); // Make sure you are accessing the correct key here
      console.log(recipe)
    } catch (error) {
      console.error('Error fetching recipe:', error);
      setRecipe('Failed to load recipe'); // Optionally handle errors in UI
    }
  };

  return (
    <div>
      <button onClick={fetchRecipe}>Get Recipe</button>
      {typeof recipe === 'string' ? (
        <p>{recipe}</p>
      ) : (
        <div>
          <p>Cannot display recipe data.</p>
          <pre>{JSON.stringify(recipe, null, 2)}</pre> 
        </div>
      )}
    </div>
  );
}

export default RecipeGen;
