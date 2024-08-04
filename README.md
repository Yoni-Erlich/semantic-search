<<<<<<< HEAD
# semantic-search
=======
# What Files are Included

	1.	yoni_artlist.ipynb: This is the main analysis notebook.
	2.	utils.py: This file contains various classes and functions, such as:
	•	CLIPEmbeddingFunction: Handles the CLIP model for embedding images and texts.
	•	ChromaHandler: Manages the ChromaDB collection, including adding embeddings, querying, and handling metadata.
	•	Additional functions for analysis and evaluation.
	3.	env.sh: Execute this script to set up the virtual environment.
	4.	requirements.txt: Lists the relevant dependencies for the project.
    5. dataset: here are the images
    6. vector_db: here wil be save the vector db

## Detailed Explanation of utils.py

The utils.py file includes essential classes and functions for handling embeddings and managing the ChromaDB collection. Below are some key components:

CLIPEmbeddingFunction

This class manages the CLIP model from the transformers library, providing methods to embed images and texts.

ChromaHandler

This class manages a ChromaDB collection, allowing you to add embeddings, query the collection, and handle metadata.

## Additional Functions

These functions help with handling query results, plotting images, and analyzing noise frequency.
	1.	get_paths_from_query_results:
Extracts the paths of images from the query results.
	2.	get_score_from_query_results:
Retrieves the scores (distances) from the query results.
	3.	plot_images:
Plots images along with their scores and names from the query results.
	4.	get_query_results_in_figs:
Queries the ChromaDB collection and plots the results.
	5.	load_images:
Loads images from the given paths and converts them to RGB format.
	6.	get_noise_apperance_frequecy_per_n_to_show:
Calculates the frequency of noise appearances for various numbers of results to show.
	7.	get_noise_apperance_frequecy:
Computes the noise appearance frequency for a given number of results to show.

>>>>>>> 931e8d0 (Initial commit)
