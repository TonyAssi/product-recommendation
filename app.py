import gradio as gr
from PIL import Image
from datasets import load_dataset, Dataset
import random
import numpy as np
import time

# Dataset
ds = load_dataset("tonyassi/finesse1-embeddings", split='train')


id_to_row = {row['id']: row for row in ds}
remaining_ds = None
preference_embedding = []

###################################################################################

def get_random_images(dataset, num):
    # Select 4 random indices from the dataset
    random_indices = random.sample(range(len(dataset)), num)
    
    # Get the 4 random images
    random_images = dataset.select(random_indices)
    
    # Create a new dataset with the remaining images
    remaining_indices = [i for i in range(len(dataset)) if i not in random_indices]
    new_dataset = dataset.select(remaining_indices)
    
    return random_images, new_dataset

def find_similar_images(dataset, num, embedding):
    # Ensure FAISS index exists and search for similar images
    dataset.add_faiss_index(column='embeddings')
    scores, retrieved_examples = dataset.get_nearest_examples('embeddings', np.array(embedding), k=num)
    
    # Drop FAISS index after use to avoid re-indexing
    dataset.drop_index('embeddings')

    # Extract all dataset IDs and use a set to find remaining indices
    dataset_ids = dataset['id']
    retrieved_ids_set = set(retrieved_examples['id'])

    # Use a list comprehension with enumerate for faster indexing
    remaining_indices = [i for i, id in enumerate(dataset_ids) if id not in retrieved_ids_set]

    # Create a new dataset without the retrieved images
    new_dataset = dataset.select(remaining_indices)

    return retrieved_examples, new_dataset

def average_embedding(embedding1, embedding2):
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    return (embedding1 + embedding2) / 2

###################################################################################

def load_images():
    print("ds", ds.num_rows)

    global remaining_ds
    remaining_ds = ds

    global preference_embedding
    preference_embedding = []

    # Get random images
    rand_imgs, remaining_ds = get_random_images(ds, 10)

    # Create a list of tuples [(img1,caption1),(img2,caption2)...]
    result = list(zip(rand_imgs['image'], [str(id) for id in rand_imgs['id']]))

    return result


def select_image(evt: gr.SelectData, gallery, preference_gallery):
    global remaining_ds
    print("remaining_ds", remaining_ds.num_rows)
    
    # Selected image
    selected_id = int(evt.value['caption'])
    selected_row = id_to_row[selected_id]
    selected_embedding = selected_row['embeddings']
    selected_image = selected_row['image']

    # Update preference embedding
    global preference_embedding
    if len(preference_embedding) == 0:
        preference_embedding = selected_embedding
    else: 
        preference_embedding = average_embedding(preference_embedding, selected_embedding)

    # Find images which are most similar to the preference embedding
    simlar_images, remaining_ds = find_similar_images(remaining_ds, 5, preference_embedding)

    # Create a list of tuples [(img1,caption1),(img2,caption2)...]
    result = list(zip(simlar_images['image'], [str(id) for id in simlar_images['id']]))

    # Get random images
    rand_imgs, remaining_ds = get_random_images(remaining_ds, 5)
    # Create a list of tuples [(img1,caption1),(img2,caption2)...]
    random_result = list(zip(rand_imgs['image'], [str(id) for id in rand_imgs['id']]))

    final_result = result + random_result

    # Update prefernce gallery
    if (preference_gallery==None):
        final_preference_gallery = [selected_image]
    else:
        final_preference_gallery = [selected_image] + preference_gallery

    return gr.Gallery(value=final_result, selected_index=None), final_preference_gallery

###################################################################################

with gr.Blocks() as demo:
    gr.Markdown("""
    <center><h1> Product Recommendation using Image Similarity </h1></center>

    <center>by <a href="https://www.tonyassi.com/" target="_blank">Tony Assi</a></center>


    <center> This is a demo of product recommendation using image similarity of user preferences. </center>

    The the user selects their favorite product which then gets added to the user preference group. Each of the image embeddings in the user preference products get averaged into a preference embedding. Each round some products are displayed: 5 products most similar to user preference embedding and 5 random products. Embeddings are generated with [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224). The dataset used is [tonyassi/finesse1-embeddings](https://huggingface.co/datasets/tonyassi/finesse1-embeddings).
    """)

    product_gallery = gr.Gallery(columns=5, object_fit='contain', allow_preview=False, label='Products')
    preference_gallery = gr.Gallery(columns=5, object_fit='contain', allow_preview=False, label='Preference', interactive=False)

    demo.load(load_images, inputs=None, outputs=[product_gallery])
    product_gallery.select(select_image, inputs=[product_gallery, preference_gallery], outputs=[product_gallery, preference_gallery])
  

demo.launch()