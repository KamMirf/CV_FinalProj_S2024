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
      const data = await response.text();
      console.log(data);

      setRecipe(data);
    } catch (error) {
      setRecipe('Failed to load recipe'); 
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
